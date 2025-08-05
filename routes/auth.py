from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('sign_up.html')
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        existing_user = User.get_by_email(email)
        if existing_user:
            flash('Email already exists')
            return redirect(url_for('auth.signup'))

        password_hash = generate_password_hash(password, method='sha256')
        User.create(username, email, password_hash)
        return redirect(url_for('auth.login'))

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('sign_in.html')
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        user = User.get_by_email(email)

        if not user or not check_password_hash(user['password_hash'], password):
            flash('Please check your login details and try again.')
            return redirect(url_for('auth.login'))

        # Create a user-like object for Flask-Login
        user_obj = type('User', (), {
            'id': user['id'],
            'is_authenticated': True,
            'is_active': True,
            'is_anonymous': False,
            'get_id': lambda: str(user['id'])
        })()

        login_user(user_obj, remember=remember)
        return redirect(url_for('main.index'))

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))
