# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:16:48 2017

@author: Michal
"""
from flask import Flask

app = Flask(__name__)

@app.route('/users/<string:username>')
def hello_world(username=None):
    return("Hello {}!".format(username))