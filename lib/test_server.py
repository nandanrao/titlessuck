import os
import server
import unittest
import tempfile
import json


def test_init():
    client = server.app.test_client()
    v = client.get('/init/1')
    assert '1' in v.data

def test_describe():
    client = server.app.test_client()
    v = client.post('/describe/1',
                    data = json.dumps({'description': 'exciting work hard all day'}),
                    content_type='application/json')
    r = json.loads(v.data)
    assert type(r["description"]) == unicode
    assert type(r["jobId"]) == str


def test_rate():
    client = server.app.test_client()
    v = client.post('/rate/1',
                    data = json.dumps({'jobId': 1, 'rating': 5}),
                    content_type='application/json')
    r = json.loads(v.data)
    assert type(r["description"]) == unicode
    assert type(r["jobId"]) == str
