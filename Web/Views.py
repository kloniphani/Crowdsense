"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import Flask, render_template, send_from_directory     


app = Flask(__name__)

import sys, os, socket, platform
import uuid, cpuinfo, psutil
import math

@app.route('/')
@app.route('/home')
def home():
    name = socket.gethostname()
    processor = cpuinfo.get_cpu_info()['brand_raw']
    ram = math.ceil((psutil.virtual_memory().total)/pow(1024, 3))
    device_id = uuid.uuid1()
    os = platform.system()

    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home',
        name = name, processor = processor, ram = ram, device_id = device_id, os = os,
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )


@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')