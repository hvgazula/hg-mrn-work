#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 19:10:22 2018

@author: Harshvardhan
"""

import os

folder_name = 'coinstac-fsl-test-data'
site_list = sorted(os.listdir(folder_name))

with open('volumes_list.txt', 'r') as volfh:
    volumes = volfh.read().splitlines()

for index, site in enumerate(site_list):
    inputspec_list = []
    folder_string = os.path.join(folder_name, site)

    all_files = os.listdir(folder_string)
    curr_csv_file = [file for file in all_files if file.endswith('csv')]

    with open(os.path.join(folder_string, curr_csv_file[0]), 'r') as csvfh:
        csv_contents = csvfh.read().splitlines()

    csv_contents = csv_contents[1:]

    with open(
            os.path.join(folder_string, 'inputspec{:02d}.json'.format(index)),
            'w') as fh:
        fh.write('''{
  "covariates": {
    "value": [
      [
[["freesurferfile", "isControl", "age"],
''')
        for curr_line in csv_contents:
            line = curr_line.split(',')
            fh.write("[\"%s\", %s, %d],\n" % (line[0], line[1].lower(),
                                              float(line[2])))
        fh.write(''' ] \n
      ],
      [
        "isControl",
        "age"
      ],
      [
        "boolean",
        "number"
      ]
    ]
  },
  "data": {
    "value": [
      [ ''')
        for curr_line in csv_contents:
            fh.write("\"%s\",\n" % curr_line.split(',')[0])
        fh.write('''  ],
      [
        "freesurferfile"
      ],
      [''')
        for region in volumes:
            fh.write("\"%s\",\n" % region)

        fh.write(''']
    ]
  },
  "lambda": {
    "value": 0
  }
}''')
