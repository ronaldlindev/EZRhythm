from re import template
from flask import Blueprint, render_template
from .functions import getfile
from datetime import date
import time
views = Blueprint('views',__name__)


@views.route('/')
def home():
    return render_template('home.html')

@views.route('/instructions')
def instructions():
    return render_template('instructions.html')
@views.route('/upload')
def upload():
    return render_template('upload.html')
@views.route('bruh')
def display():
    time.sleep(20)
    # getfile()
    return render_template('display.html', Patient_Name = 'Ronald Lin', Gender = 'Male', date = date.today(),dob = '6/22/05', age = 16, maxhr = 130, minhr = 52, averagehr = 55, primarydiagnosis = 'Normal ECG! Congrats!', sessionlen = '12:24:10', sessionid = 759425)
