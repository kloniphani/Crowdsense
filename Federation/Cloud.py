import pyrebase
from pyrebase.pyrebase import Database

class Google():
    # ----------------------------
    # Firebase Google cloud
    # ----------------------------

    firebase_config = {
        'apiKey': "AIzaSyAUwC9sGdHEhNJZxTX3BJL_7L2HU0FzbmM",
        'authDomain': "pollution-fcc8d.firebaseapp.com",
        'databaseURL': "https://pollution-fcc8d-default-rtdb.firebaseio.com",
        'projectId': "pollution-fcc8d",
        'storageBucket': "pollution-fcc8d.appspot.com",
        'messagingSenderId': "750667679514",
        'appId': "1:750667679514:web:bfb3d06a066eed326f4ea3",
        'measurementId': "G-2N1WPMPYFX"
    }

    def __init__(self):
        self.firebase = pyrebase.initialize_app(self.firebase_config)
        self.database = self.firebase.database()

