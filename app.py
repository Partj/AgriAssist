import os
from flask import Flask, render_template, request, redirect, flash, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
# Fetch the cloud database URL if available, otherwise use local SQLite
db_url = os.environ.get('DATABASE_URL', 'sqlite:///agriassist.db')

# SQLAlchemy requires 'postgresql://', but some cloud providers give 'postgres://'. This fixes that crash.
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'a_very_secret_key_for_security'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

rf_model = joblib.load('crop_recommender.joblib')
xgb_model = joblib.load('yield_predictor.joblib')
encoder = joblib.load('crop_encoder.joblib')

# --- DATABASE MODELS ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    city = db.Column(db.String(100), default="Delhi")
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

class Scheme(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    link = db.Column(db.String(300), nullable=False)

# NEW: Track AI History
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pred_type = db.Column(db.String(50), nullable=False) # 'Crop' or 'Yield'
    result_text = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def get_weather_data(city_name):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
        geo_response = requests.get(geo_url).json()
        if "results" in geo_response:
            lat = geo_response["results"][0]["latitude"]
            lon = geo_response["results"][0]["longitude"]
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,precipitation"
            weather_response = requests.get(weather_url).json()
            temp = weather_response["current"]["temperature_2m"]
            rain = weather_response["current"]["precipitation"]
            if rain < 10: rain = 120.0
            return temp, rain
    except Exception as e:
        pass
    return 28.0, 100.0

def get_current_season():
    m = datetime.now().month
    if m in [6, 7, 8, 9, 10]: return "Kharif"
    elif m in [11, 12, 1, 2, 3]: return "Rabi"
    return "Zaid"

# --- AUTH ROUTES ---
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        phone = request.form.get("phone")
        password = request.form.get("password")
        if User.query.filter_by(phone=phone).first():
            flash("Phone number already registered. Please log in.")
            return redirect(url_for('login'))
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(name=name, phone=phone, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        phone = request.form.get("phone")
        password = request.form.get("password")
        user = User.query.filter_by(phone=phone).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash("Invalid phone number or password.")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- NEW UX ARCHITECTURE ROUTES ---

@app.route("/")
@login_required
def home():
    # Pure Display Route
    user_entries = Khata.query.filter_by(user_id=current_user.id).all()
    user_crops = MyCrop.query.filter_by(user_id=current_user.id).all()
    recent_preds = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).limit(4).all()
    dash_temp, _ = get_weather_data(current_user.city)
    
    total_income = sum(e.amount for e in user_entries if e.transaction_type == 'Income')
    total_expense = sum(e.amount for e in user_entries if e.transaction_type == 'Expense')
    net_balance = total_income - total_expense
    total_acres = sum(c.acres for c in user_crops)

    return render_template("index.html", 
                           entries=user_entries[-5:], # Show only last 5 on dashboard
                           crops=user_crops, 
                           predictions=recent_preds,
                           weather={"temperature": dash_temp},
                           stats={"income": total_income, "expense": total_expense, "balance": net_balance, "acres": total_acres, "season": get_current_season()})

@app.route("/khata", methods=["GET", "POST"])
@login_required
def khata():
    if request.method == "POST":
        item = request.form.get("item_name")
        amt = request.form.get("amount")
        t_type = request.form.get("transaction_type")
        new_entry = Khata(item_name=item, amount=amt, transaction_type=t_type, user_id=current_user.id)
        db.session.add(new_entry)
        db.session.commit()
        return redirect(url_for('khata'))
    
    entries = Khata.query.filter_by(user_id=current_user.id).all()
    return render_template("khata.html", entries=entries[::-1]) # Show newest first

@app.route("/delete_khata/<int:entry_id>", methods=["POST"])
@login_required
def delete_khata(entry_id):
    entry = Khata.query.get(entry_id)
    if entry and entry.user_id == current_user.id:
        db.session.delete(entry)
        db.session.commit()
    return redirect(url_for('khata'))

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "update_npk":
            current_user.city = request.form.get("city")
            current_user.n_level = float(request.form.get("n_level"))
            current_user.p_level = float(request.form.get("p_level"))
            current_user.k_level = float(request.form.get("k_level"))
            db.session.commit()
            flash("Profile updated successfully!")
        elif action == "add_crop":
            crop_name = request.form.get("crop_name")
            acres = float(request.form.get("acres"))
            new_crop = MyCrop(crop_name=crop_name, acres=acres, user_id=current_user.id)
            db.session.add(new_crop)
            db.session.commit()
            flash(f"Added {crop_name} to your farm.")
        return redirect(url_for('profile'))
    
    user_crops = MyCrop.query.filter_by(user_id=current_user.id).all()
    return render_template("profile.html", crops=user_crops)

@app.route("/delete_crop/<int:crop_id>", methods=["POST"])
@login_required
def delete_crop(crop_id):
    crop = MyCrop.query.get(crop_id)
    if crop and crop.user_id == current_user.id:
        db.session.delete(crop)
        db.session.commit()
    return redirect(url_for('profile'))

# --- AI & TOOLS ---
@app.route("/recommend", methods=["GET", "POST"])
@login_required
def recommend():
    result = None
    if request.method == "POST":
        n, p, k = float(request.form.get("nitrogen")), float(request.form.get("phosphorus")), float(request.form.get("potassium"))
        city = request.form.get("city")
        temp, rain = get_weather_data(city)
        season = get_current_season()
        
        input_data = pd.DataFrame([[n, p, k, temp, rain]], columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Rainfall'])
        probabilities = rf_model.predict_proba(input_data)[0]
        crop_names = rf_model.classes_ 
        crop_scores = sorted(list(zip(crop_names, probabilities)), key=lambda x: x[1], reverse=True)
        
        top_crops = [{"name": c, "confidence": round(pr * 100, 1)} for c, pr in crop_scores[:3] if pr > 0.05]
        
        # Save to History
        if top_crops:
            new_pred = Prediction(user_id=current_user.id, pred_type="Crop Recommendation", result_text=f"Top Match: {top_crops[0]['name']} ({top_crops[0]['confidence']}%)")
            db.session.add(new_pred)
            db.session.commit()

        result = {"top_crops": top_crops, "city": city, "temp": temp, "rain": rain, "season": season}
    return render_template("recommend.html", result=result)

@app.route("/yield", methods=["GET", "POST"])
@login_required
def predict_yield():
    result = None
    if request.method == "POST":
        n, p, k = float(request.form.get("nitrogen")), float(request.form.get("phosphorus")), float(request.form.get("potassium"))
        city, crop_name = request.form.get("city"), request.form.get("crop")
        temp, rain = get_weather_data(city)
        crop_numeric = encoder.transform([crop_name])[0]
        
        input_data = pd.DataFrame([[n, p, k, temp, rain, crop_numeric]], columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Rainfall', 'Crop_Numeric'])
        predicted_yield = round(xgb_model.predict(input_data)[0], 2)
        
        # Save to History
        new_pred = Prediction(user_id=current_user.id, pred_type="Yield Forecast", result_text=f"{crop_name} in {city}: {predicted_yield} kg/acre")
        db.session.add(new_pred)
        db.session.commit()

        result = {"yield": predicted_yield, "crop": crop_name, "city": city}
    return render_template("yield.html", result=result)

@app.route("/fertilizer", methods=["GET", "POST"])
@login_required
def fertilizer():
    result = None 
    if request.method == "POST":
        crop, acres = request.form.get("crop"), float(request.form.get("acres"))
        if crop == "Wheat": std_n, std_p, std_k = 50, 25, 12
        elif crop == "Rice": std_n, std_p, std_k = 60, 30, 20
        else: std_n, std_p, std_k = 40, 20, 10
            
        needed_n, needed_p, needed_k = max(0, std_n - current_user.n_level), max(0, std_p - current_user.p_level), max(0, std_k - current_user.k_level)
        result = {
            "crop": crop, "acres": acres, "urea": round(needed_n * acres, 2), "dap": round(needed_p * acres, 2), "mop": round(needed_k * acres, 2),
            "adjusted": (current_user.n_level + current_user.p_level + current_user.k_level) > 0
        }
    return render_template("fertilizer.html", result=result)

@app.route("/schemes")
@login_required
def schemes():
    search_query = request.args.get('search', '')
    if search_query:
        all_schemes = Scheme.query.filter((Scheme.name.ilike(f'%{search_query}%')) | (Scheme.description.ilike(f'%{search_query}%'))).all()
    else:
        all_schemes = Scheme.query.all()
    return render_template("schemes.html", schemes=all_schemes, search_query=search_query)

with app.app_context():
    db.create_all()
    if not Scheme.query.first():
        default_schemes = [
            Scheme(name="PM-KISAN (Direct Income Support)", description="Financial benefit of ₹6,000 per year.", link="https://pmkisan.gov.in/"),
            Scheme(name="PMFBY (Crop Insurance)", description="Insurance for crops against natural calamities.", link="https://pmfby.gov.in/"),
            Scheme(name="Kisan Credit Card (KCC)", description="Access to short-term credit for crops.", link="https://sbi.co.in/web/agri-rural/agriculture-banking/crop-loan/kisan-credit-card")
        ]
        db.session.bulk_save_objects(default_schemes)
        db.session.commit()

if __name__ == "__main__":
    app.run(debug=True)