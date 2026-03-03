import os
import requests
from datetime import datetime
from flask import Flask, render_template, request, redirect, flash, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# --- CONFIGURATION ---
# Replace with your actual Hugging Face API URL once it is running
HF_API_URL = "https://yourusername-agriassist-api.hf.space" 

# Database logic: Use Supabase in production, SQLite for local testing
db_url = os.environ.get('DATABASE_URL', 'sqlite:///agriassist.db')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'a_very_secret_key_for_security'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# --- DATABASE MODELS ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    city = db.Column(db.String(100), default="Chennai")
    n_level = db.Column(db.Float, default=0.0)
    p_level = db.Column(db.Float, default=0.0)
    k_level = db.Column(db.Float, default=0.0)
    khata_entries = db.relationship('Khata', backref='user', lazy=True)
    my_crops = db.relationship('MyCrop', backref='user', lazy=True)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Khata(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_name = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Integer, nullable=False)
    transaction_type = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class MyCrop(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    crop_name = db.Column(db.String(50), nullable=False)
    acres = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pred_type = db.Column(db.String(50), nullable=False) 
    result_text = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))





# --- HELPER FUNCTIONS ---
def get_weather_data(city_name):
    # Mock weather for demo stability, or use an API like OpenWeather
    return 28.0, 85.0, 6.5, 120.0 # temp, hum, ph, rain

@app.route("/setup-db")
def setup_db():
    try:
        db.create_all()
        return "Database tables created successfully! You can now go to /register."
    except Exception as e:
        return f"Database creation failed: {str(e)}"

# --- ROUTES ---

@app.route("/")
@login_required
def home():
    # 1. Get database records
    user_crops = MyCrop.query.filter_by(user_id=current_user.id).all()
    recent_preds = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).limit(4).all()
    
    # 2. Package the stats data
    stats = {
        "season": "Kharif", 
        "active_crops": len(user_crops)
    }

    # 3. Package the weather data using your helper function
    temp, hum, ph, rain = get_weather_data(current_user.city)
    weather = {
        "temperature": temp,
        "humidity": hum,
        "rainfall": rain
    }

    # 4. Deliver ALL variables to the HTML template
    return render_template("index.html", 
                           crops=user_crops, 
                           predictions=recent_preds, 
                           stats=stats, 
                           weather=weather)
# --- THE MICROSERVICE AI ROUTES ---

@app.route("/recommend", methods=["GET", "POST"])
@login_required
def recommend():
    result = None
    if request.method == "POST":
        n = float(request.form.get("nitrogen"))
        p = float(request.form.get("phosphorus"))
        k = float(request.form.get("potassium"))
        city = request.form.get("city")
        temp, hum, ph, rain = get_weather_data(city)

        # Send to Hugging Face
        payload = {"features": [n, p, k, temp, hum, ph, rain]}
        try:
            response = requests.post(f"{HF_API_URL}/predict_crop", json=payload, timeout=15)
            top_crops = response.json().get('result', [])
            
            if top_crops:
                new_pred = Prediction(
                    user_id=current_user.id, 
                    pred_type="Crop Recommendation", 
                    result_text=f"Top Match: {top_crops[0]['name']}"
                )
                db.session.add(new_pred)
                db.session.commit()
            result = {"top_crops": top_crops}
        except Exception as e:
            flash("AI Service is warming up. Please wait 30 seconds and try again.")
            print(f"API Error: {e}")

    return render_template("recommend.html", result=result)

@app.route("/yield", methods=["GET", "POST"])
@login_required
def predict_yield():
    result = None
    if request.method == "POST":
        # In a real app, this comes from a dropdown matching your label encoder
        crop_num = int(request.form.get("crop_numeric", 22)) 
        n, p, k = current_user.n_level, current_user.p_level, current_user.k_level
        temp, hum, ph, rain = get_weather_data(current_user.city)

        # Send to Hugging Face
        payload = {"features": [n, p, k, temp, hum, rain, crop_num]}
        try:
            response = requests.post(f"{HF_API_URL}/predict_yield", json=payload, timeout=15)
            predicted_yield = response.json().get('yield')
            
            new_pred = Prediction(
                user_id=current_user.id, 
                pred_type="Yield Forecast", 
                result_text=f"Predicted Yield: {predicted_yield} tonnes/hectare"
            )
            db.session.add(new_pred)
            db.session.commit()
            result = {"yield": predicted_yield}
        except Exception as e:
            flash("AI Service Error or waking up.")
            print(f"API Error: {e}")

    return render_template("yield.html", result=result)


# --- AUTH & PROFILE ROUTES ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(phone=request.form.get("phone")).first()
        if user and check_password_hash(user.password, request.form.get("password")):
            login_user(user)
            return redirect(url_for('home'))
        flash("Invalid Credentials")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Check if user already exists
        existing_user = User.query.filter_by(phone=request.form.get("phone")).first()
        if existing_user:
            flash("Phone number already registered. Please log in.")
            return redirect(url_for('login'))
            
        hashed_pw = generate_password_hash(request.form.get("password"), method='pbkdf2:sha256')
        new_user = User(
            name=request.form.get("name"), 
            phone=request.form.get("phone"), 
            password=hashed_pw,
            city=request.form.get("city", "Chennai") # Default fallback
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please log in.")
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        current_user.n_level = float(request.form.get("n_level"))
        current_user.p_level = float(request.form.get("p_level"))
        current_user.k_level = float(request.form.get("k_level"))
        current_user.city = request.form.get("city")
        db.session.commit()
        flash("Profile Updated Successfully")
    return render_template("profile.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Render is a real server, so we can do this normally!
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)