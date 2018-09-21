#!/usr/bin/env python

from subprocess import call
import os
import time

# We want to start a server from each www directory
# where everything was built by the site-builder script

# Make sure jupyter defaults are correct (globally)

call("jupyter nbextension enable hide_input/main", shell=True)
call("jupyter nbextension enable rubberband/main", shell=True)
call("jupyter nbextension enable exercise/main", shell=True)

# This could be automated, but I am not sure how well the number of
# servers will scale ... so leave at 8 ... and hand build

# The root user is www

users   = { "www"       : ["vieps-pye-boss", 8080 ],
            "build/www1": ["vieps-pye-1",    8081 ],
            "build/www2": ["vieps-pye-2",    8082 ],
            "build/www3": ["vieps-pye-3",    8083 ],
            "build/www4": ["vieps-pye-4",    8084 ],
            "build/www5": ["vieps-pye-5",    8085 ],
            "build/www6": ["vieps-pye-6",    8086 ],
            "build/www7": ["vieps-pye-7",    8087 ],
            "build/www8": ["vieps-pye-8",    8088 ],
            "build/www9": ["vieps-pye-9",    8089 ],
            "build/www10": ["vieps-pye-10",  8090 ],
            "build/www11": ["vieps-pye-11",  8091 ],
            "build/www12": ["vieps-pye-12",  8092 ] }

# Maybe we need to quote the password in case it has odd characters in it

for dir in users.keys():
    password = users[dir][0]
    port     = users[dir][1]
    call( "cd {:s} && nohup jupyter notebook --port={:d} --ip='*' --no-browser\
           --NotebookApp.token={:s} --NotebookApp.default_url=/files/index.html &".format(dir, port, password), shell=True )

# Don't exit

while True:
    time.sleep(10)
