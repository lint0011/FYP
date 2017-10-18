# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:01:12 2017

@author: lint0011
"""

import MySQLdb as mdb
import MySQLdb.cursors
import re
import csv

#connect to information_scheme database to get the size of the table test_tags
get_test_tags_size = mdb.connect('localhost', 'root','l56530304T','information_schema', cursorclass=MySQLdb.cursors.SSCursor);
cur = get_test_tags_size.cursor()

#get table size
table_name = 'test_tags'
get_size = 'SELECT TABLE_ROWS FROM TABLES WHERE TABLE_NAME = \''+ table_name + '\''
cur.execute(get_size)
table_size = cur.fetchall()[0][0]
print('The size of table test_tags is ' + str(table_size))

#connect to test database to execute query on the table of test_tags
con = mdb.connect('localhost', 'root','l56530304T','test',cursorclass=MySQLdb.cursors.SSCursor);
cur = con.cursor()

#fetch all rows in test_tags to perform cleaning on TagName
select_distinct_tags_and_counts = 'SELECT DISTINCT TagName,Count FROM test_tags\
                        WHERE TagName REGEXP \'[0-9]\' \
                        '
cur.execute(select_distinct_tags_and_counts)

#special terms that need to be filtered out when performing cleaning
#A list of list, in the format of ['original term', 'designated term']
special_terms = [['amazon-ec2','amazon-ec2'], ['amazon-s3','amazon-s3'],
                 ['c++1z','c++'],['x11','x11'],['t4','t4'],
                 ['x86','x86'],['x86-64','x86'],['base64','base64'],
                 ['mp3','mp3'],['mp4','mp4'],['v8','v8'],
                 ['flex3','flex'],['h2','h2'],['python-2.x','python'],
                 ['8086','8086'], ['h.264', 'h.264'],['flex4','flex'],
                 ['x509','x509'],['flex4.5','flex'],['z3','z3'],
                 ['opencv3.0','opencv']
                 ]


#crete a list of list to be printed into a csv file for study
original_cleaned_count = []

#for all distinct tags fetched from test_tags, clean and output to csv file
for row in cur:
    
    if row[0] is not None:
        
        count = row[1]
        original_term = row[0]
        temp_term = row[0]
        
        #firstly, check whether it is one of the special terms:
        for term in special_terms:
            #if indeed is a special term, then change the format according to the designated term
            if original_term == term[0]:
                cleaned_term = term[1]
                break
        
        #if it is not a special term
        else:
            #check whether it ends with 'N(.Nxrv)'
            #need to check multiple times because there can be format of 'python-3.6-v2'
            while bool(re.search('[xvr]*[0-9.]+$',temp_term)) :
                if bool(re.search('-[xvr]*[0-9.]+$',temp_term)):
                    temp_term = re.sub('-[xvr]*[0-9.]+$','',temp_term)
                else:
                    temp_term = re.sub('[xvr]*[0-9.]+$','',temp_term)
            
            #at the end of the while loop, the temp_term doesn't contain any uncleaned format
            cleaned_term = temp_term
        
        original_cleaned_count.append([original_term,cleaned_term,count]) 
        #print('The original term is '+original_term+', the cleaned term is '+cleaned_term+', the count is '+str(row[1]))

#print original_cleaned_count to a csv file
csvfile = "D:\FYP\cleaned_result_from_v1.1.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(original_cleaned_count)
                    
        

                        
                        


