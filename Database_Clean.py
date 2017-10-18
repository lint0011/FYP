# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:24:20 2017

@author: lint0011
"""

#the following code is used to cleaned the format of Tags in database test_post, 
#such that all the representation of different versions is eliminated

import MySQLdb as mdb
import MySQLdb.cursors
import re
import csv

#firstly, extract Tags, Id from each rows in the databases;
#for testing purpose, limit to DESC 200;

#connect to database:
con = mdb.connect('localhost', 'root','l56530304T','test',cursorclass=MySQLdb.cursors.SSCursor);
cur = con.cursor()

#fetch the last 200 rows in test_post that contains [0-9], to perform tags cleaning
select_Id_and_Tags = 'SELECT Id,Tags FROM test_post\
                        WHERE Tags REGEXP \'[0-9]\' \
                        '
cur.execute(select_Id_and_Tags)

#special terms that need to be filtered out when performing cleaning
#A list of list, in the format of ['original term', 'designated term']
special_terms = [['amazon-ec2','amazon-ec2'], ['amazon-s3','amazon-s3'],
                 ['c++1z','c++'],['x11','x11'],['t4','t4'],
                 ['x86','x86'],['x86-64','x86'],['base64','base64'],
                 ['mp3','mp3'],['mp4','mp4'],['v8','v8'],
                 ['flex3','flex'],['h2','h2'],['python-2.x','python'],
                 ['8086','8086'], ['h.264', 'h.264'],['flex4','flex'],
                 ['x509','x509'],['flex4.5','flex'],['z3','z3'],
                 ['opencv3.0','opencv'],['python-3.x','python'],
                 ['opencv3.2','opencv']
                 ]

#array to host rows that need to be updated:
updated_rows = []

#each row is in the format of : (45901665, '<spring-security><oauth-2.0>')
for row in cur:
    #first of all, check Tags is not None:
    if row[1] is not None:
        
        Id = row[0]
        original_tags = row[1]
        #indicate whether we need to update the database
        changed = False
        
        #new_row of updated_rows:
        new_row = []
        
        #split Tags into individual tags:
        subbed_tags = re.sub('<|>',' ',original_tags)
        split_tags = subbed_tags.split()
        
        #for each tag, perform tag cleaning:
        for i, tag in enumerate(split_tags):
            
            original_tag= tag
            temp_tag = tag
        
            #firstly, check whether it is one of the special terms:
            for term in special_terms:
                #if indeed is a special term, then change the format according to the designated term
                if tag == term[0]:
                    cleaned_tag = term[1]
                    break
       
            #if it is not a special term
            else:
                #check whether it ends with 'N(.Nxrv)'
                #need to check multiple times because there can be format of 'python-3.6-v2'
                while bool(re.search('[xvr]*[0-9.]+$',temp_tag)) :
                    if bool(re.search('-[xvr]*[0-9.]+$',temp_tag)):
                        temp_tag = re.sub('-[xvr]*[0-9.]+$','',temp_tag)
                    else:
                        temp_tag = re.sub('[xvr]*[0-9.]+$','',temp_tag)
                
                #at the end of the while loop, the temp_tag doesn't contain any uncleaned format
                cleaned_tag = temp_tag
                
            #if cleaned_tag is not the same as the original_tag, then need to updte tag in split_tags
            if cleaned_tag != original_tag:
                print(cleaned_tag+' '+original_tag+' ' + str(Id))
                split_tags[i] = cleaned_tag
                changed = True
        
        #now that all the tags are cleaned, put them back into the format of '<python><python-3.x><pickle><python-multiprocessing>'
        if changed:
            #if there are changes that need to be updated:
            cleaned_tags = '<'+"><".join(split_tags)+'>'
            #update Id:
            new_row.append(Id)
            #update Tags:
            new_row.append(cleaned_tags)
            #append new_row to updated_row:
            updated_rows.append(new_row)
            

#print Id_cleaned_tag to a csv file
csvfile = "D:\FYP\Id_cleaned_tag_from_Database_Clean_v1.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(updated_rows)


