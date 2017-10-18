# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:44:55 2017

@author: lint0011
"""

import MySQLdb as mdb
import MySQLdb.cursors
import csv

#connect to database:
con = mdb.connect('localhost', 'root','l56530304T','test',cursorclass=MySQLdb.cursors.SSCursor);
cur = con.cursor()

#fetch Body in test_post with PostTypeId = 4
select_Id_and_Tags = " SELECT Id, Body FROM test_post WHERE PostTypeId = 4"
cur.execute(select_Id_and_Tags)

for row in cur:
    print(row)

#print Id_cleaned_tag to a csv file
csvfile = "D:\FYP\Id_Body_of_TagWiki_Excerpt.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(cur)