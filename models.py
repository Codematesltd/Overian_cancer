from datetime import datetime
from supabase import create_client
from config import Config

# Initialize Supabase client
supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

class User:
    @staticmethod
    def create(username, email, password_hash):
        user_data = {
            'username': username,
            'email': email,
            'password_hash': password_hash
        }
        return supabase.table('users').insert(user_data).execute()

    @staticmethod
    def get_by_email(email):
        response = supabase.table('users').select('*').eq('email', email).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def get_by_id(user_id):
        response = supabase.table('users').select('*').eq('id', user_id).execute()
        return response.data[0] if response.data else None

class Prediction:
    @staticmethod
    def create(user_id, age, ca125_level, tumor_size, prediction_result):
        prediction_data = {
            'user_id': user_id,
            'age': age,
            'ca125_level': ca125_level,
            'tumor_size': tumor_size,
            'prediction_result': prediction_result,
            'timestamp': datetime.utcnow().isoformat()
        }
        return supabase.table('predictions').insert(prediction_data).execute()

    @staticmethod
    def get_user_predictions(user_id):
        return supabase.table('predictions')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('timestamp', desc=True)\
            .execute()
