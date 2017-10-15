# FYP

FYP Notes

Difference btw varchar and longtext: https://stackoverflow.com/questions/25300821/difference-between-varchar-and-text-in-mysql

Select distinct from …
Cd python3.0>python –m pip install genism
Python command line: python [-bBdEiOQsRStuUvVWxX3?] [-c command | -m module-name | script | - ] [args]
Word2Vec tutorial I used: 
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
https://rare-technologies.com/word2vec-tutorial/#word2vec_tutorial
Tutorial to use pandas library: http://nbviewer.jupyter.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/01%20-%20Lesson.ipynb
CSV: comma separated value
Export mysql queries to csv file: http://www.mysqltutorial.org/mysql-export-table-to-csv/
mysql> SELECT
    -> IFNULL(Tags, 'N/A')
    -> FROM post_askubuntu
    -> INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.7/Uploads/tags.csv';
Query OK, 11952 rows affected (0.21 sec)
mysql> use fyp
Database changed
mysql> SELECT Tags FROM post_askubuntu
    -> WHERE Tags IS NOT NULL
    -> INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.7/Uploads/tags.csv'
    -> FIELDS ENCLOSED BY '"';
Query OK, 4688 rows affected (0.22 sec)
Import Tags CSV into python using pandas:
>>> import pandas as pd
>>> location = r'E:\fyp\word2vec tutorial\tags.csv'
>>> df = pd.read_csv(location, names = ['Tags'])
>>> df
                                                   Tags
0                  <discussion><top-7><site-attributes>
1                                  <discussion><applet>
2                                   <discussion><scope>
3                      <discussion><scope><derivatives>
4                             <discussion><top-7><meta>
5                                    <discussion><tags>
6                                   <discussion><scope>
7                                 <help><stackexchange>
8                  <discussion><status-completed><tags>
9                                    <discussion><tags>
10                                  <discussion><scope>
11                                  <discussion><scope>
12                        <bug><status-bydesign><users>
13                          <discussion><answer-format>
14    <feature-request><status-bydesign><voting><clo...
15                                         <discussion>
16                                         <discussion>
17            <discussion><top-7><moderation><election>
18                                  <discussion><tools>
19                                  <discussion><scope>
20                                  <discussion><scope>
21                 <discussion><top-7><site-attributes>
22                                  <discussion><top-7>
23          <discussion><answer-format><faq-suggestion>
24                             <discussion><moderation>
25                                         <discussion>
26                 <discussion><top-7><site-attributes>
27          <discussion><top-7><site-attributes><style>
28                                         <discussion>
29                                <discussion><answers>
...                                                 ...
4658          <discussion><scope><third-party-software>
4659                                   <support><login>
4660                                       <discussion>
4661                           <discussion><duplicates>
4662                                       <discussion>
4663                              <discussion><answers>
4664                              <discussion><cleanup>
4665                                          <support>
4666                                  <feature-request>
4667                                       <discussion>
4668                    <support><reputation><up-votes>
4669                                          <support>
4670  <discussion><answers><duplicates><review><unan...
4671                                       <discussion>
4672              <support><bounty><answered-questions>
4673                              <support><formatting>
4674                              <discussion><support>
4675                                 <discussion><tags>
4676                                  <feature-request>
4677                                       <discussion>
4678            <discussion><scope><voting><downvoting>
4679                                       <discussion>
4680                              <discussion><answers>
4681  <support><editing><comments><hyperlinks><spam-...
4682                <feature-request><asking-questions>
4683  <support><editing><answers><comments><moderation>
4684                                       <discussion>
4685                                       <discussion>
4686                          <discussion><end-of-life>
4687               <discussion><support><user-accounts>

[4688 rows x 1 columns]
>>>
Define the function to transfer lines in csv file into lists:
def tag_to_taglist(tags):
	tag_cleaned = re.sub("[<>]"," ", tags)
	tags_list = tag_cleaned.split()
	return tags_list
Convert dataframe into a list of lists of tags:
	>>>import re
>>> tags_group = [] #Initialize an empty list of tags_group
>>> print ("Parsing tags from post_askubuntu")
Parsing tags from post_askubuntu
>>> for tags in df["Tags"]:
	tags_group.append( tag_to_taglist(tags))
>>> print(len(tags_group))
4688
>>>
Examine if step 14 works:
>>> print(tags_group[5])
['discussion', 'tags']
>>> print(tags_group[14])
['feature-request', 'status-bydesign', 'voting', 'closed-questions']
Train Word2Vec Model
>>> import logging
>>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
>>> num_features = 200
>>> min_word_count = 4
>>> num_workers = 2
>>> context = 2
>>> downsampling = 1e-3
>>> from gensim.models import word2vec

Warning (from warnings module):
  File "D:\Python3.0\lib\site-packages\gensim\utils.py", line 865
    warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
UserWarning: detected Windows; aliasing chunkize to chunkize_serial
2017-09-06 15:06:01,145 : INFO : 'pattern' package not found; tag filters are not available for English
>>> print ("Training model...")
Training model...
>>> model = word2vec.Word2Vec(tags_group, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
2017-09-06 15:07:26,322 : INFO : collecting all words and their counts
2017-09-06 15:07:26,356 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
2017-09-06 15:07:26,413 : INFO : collected 189 word types from a corpus of 9921 raw words and 4688 sentences
2017-09-06 15:07:26,486 : INFO : Loading a fresh vocabulary
2017-09-06 15:07:26,525 : INFO : min_count=4 retains 155 unique words (82% of original 189, drops 34)
2017-09-06 15:07:26,605 : INFO : min_count=4 leaves 9852 word corpus (99% of original 9921, drops 69)
2017-09-06 15:07:26,700 : INFO : deleting the raw counts dictionary of 189 items
2017-09-06 15:07:26,750 : INFO : sample=0.001 downsamples 50 most-common words
2017-09-06 15:07:26,797 : INFO : downsampling leaves estimated 3145 word corpus (31.9% of prior 9852)
2017-09-06 15:07:26,897 : INFO : estimated required memory for 155 words and 200 dimensions: 325500 bytes
2017-09-06 15:07:27,005 : INFO : resetting layer weights
2017-09-06 15:07:27,060 : INFO : training model with 2 workers on 155 vocabulary and 200 features, using sg=0 hs=0 sample=0.001 negative=5 window=2
2017-09-06 15:07:27,245 : INFO : worker thread finished; awaiting finish of 1 more threads
2017-09-06 15:07:27,352 : INFO : worker thread finished; awaiting finish of 0 more threads
2017-09-06 15:07:27,418 : INFO : training on 49605 raw words (15716 effective words) took 0.2s, 71402 effective words/s
2017-09-06 15:07:27,538 : WARNING : under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay
>>> model.init_sims(replace=True)
2017-09-06 15:08:44,268 : INFO : precomputing L2-norms of word weight vectors
>>> model_name = "200features_4minwords_3context_for_tags_askubuntu"
>>> model.save(model_name)
2017-09-06 15:09:46,390 : INFO : saving Word2Vec object under 200features_4minwords_3context_for_tags_askubuntu, separately None
2017-09-06 15:09:46,527 : INFO : not storing attribute syn0norm
2017-09-06 15:09:46,592 : INFO : not storing attribute cum_table
2017-09-06 15:09:46,660 : INFO : saved 200features_4minwords_3context_for_tags_askubuntu

The methods to list all the functions in the module in python:
dir(word2vec)
['BrownCorpus', 'Dictionary', 'Empty', 'FAST_VERSION', 'GeneratorType', 'KeyedVectors', 'LineSentence', 'MAX_WORDS_IN_BATCH', 'PathLineSentences', 'Queue', 'REAL', 'Text8Corpus', 'Vocab', 'Word2Vec', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'array', 'ascontiguousarray', 'call_on_class_only', 'deepcopy', 'default_timer', 'defaultdict', 'division', 'dot', 'double', 'dtype', 'empty', 'exp', 'expit', 'fromstring', 'heapq', 'iteritems', 'itertools', 'itervalues', 'keep_vocab_item', 'log', 'logaddexp', 'logger', 'logging', 'matutils', 'ndarray', 'newaxis', 'np_sum', 'ones', 'os', 'outer', 'prod', 'random', 'score_cbow_pair', 'score_sentence_cbow', 'score_sentence_sg', 'score_sg_pair', 'seterr', 'sqrt', 'stats', 'string_types', 'sys', 'threading', 'train_batch_cbow', 'train_batch_sg', 'train_cbow_pair', 'train_sg_pair', 'uint32', 'uint8', 'utils', 'vstack', 'warnings', 'xrange', 'zeros']

>>> dir(word2vec.Word2Vec)
['__class__', '__contains__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slotnames__', '__str__', '__subclasshook__', '__weakref__', '_adapt_by_suffix', '_do_train_job', '_load_specials', '_minimize_model', '_raw_word_count', '_save_specials', '_smart_save', 'accuracy', 'build_vocab', 'clear_sims', 'create_binary_tree', 'delete_temporary_training_data', 'doesnt_match', 'estimate_memory', 'evaluate_word_pairs', 'finalize_vocab', 'get_latest_training_loss', 'init_sims', 'initialize_word_vectors', 'intersect_word2vec_format', 'load', 'load_word2vec_format', 'log_accuracy', 'log_evaluate_word_pairs', 'make_cum_table', 'most_similar', 'most_similar_cosmul', 'n_similarity', 'predict_output_word', 'reset_from', 'reset_weights', 'save', 'save_word2vec_format', 'scale_vocab', 'scan_vocab', 'score', 'seeded_vector', 'similar_by_vector', 'similar_by_word', 'similarity', 'sort_vocab', 'train', 'update_weights', 'wmdistance']


Test out the result:
	>>> model.doesnt_match("voting downvoting top-7".split())
'top-7'

>>> model.most_similar("scope")
[('discussion', 0.9980337619781494), ('support', 0.9979943037033081), ('bug', 0.9979202747344971), ('feature-request', 0.997847318649292), ('status-completed', 0.9977119565010071), ('design', 0.9977080821990967), ('comments', 0.9975202679634094), ('flagging', 0.9975036382675171), ('review', 0.9973804950714111), ('close-reasons', 0.9972469806671143)]


20. mysql python tutorial: http://zetcode.com/db/mysqlpython/
21. import posts.xml into database fyp
22. mysql> CREATE TABLE test_post(
    -> Id INT NOT NULL,
    -> PostTypeId INT,
    -> AcceptedAnswerId INT,
    -> ParentId INT,
    -> CreationDate DATETIME,
    -> DeletionDate DATETIME,
    -> Score INT,
    -> ViewCount INT,
    -> Body TEXT,
    -> OwnerUserId INT,
    -> OwnerDisplayName TINYBLOB,
    -> LastEditorUserId INT,
    -> LastEditorDisplayName TINYBLOB,
    -> LastEditDate DATETIME,
    -> LastActivityDate DATETIME,
    -> Title TINYTEXT,
    -> Tags TINYTEXT,
    -> AnswerCount INT,
    -> CommentCount INT,
    -> FavoriteCount INT,
    -> ClosedDate DATETIME,
    -> CommunityOwnedDate DATETIME,
    -> PRIMARY KEY (Id)
    -> );
Query OK, 0 rows affected (0.34 sec)

23. mysql> describe test_post;
+-----------------------+----------+------+-----+---------+-------+
| Field                 | Type     | Null | Key | Default | Extra |
+-----------------------+----------+------+-----+---------+-------+
| Id                    | int(11)  | NO   | PRI | NULL    |       |
| PostTypeId            | int(11)  | YES  |     | NULL    |       |
| AcceptedAnswerId      | int(11)  | YES  |     | NULL    |       |
| ParentId              | int(11)  | YES  |     | NULL    |       |
| CreationDate          | datetime | YES  |     | NULL    |       |
| DeletionDate          | datetime | YES  |     | NULL    |       |
| Score                 | int(11)  | YES  |     | NULL    |       |
| ViewCount             | int(11)  | YES  |     | NULL    |       |
| Body                  | text     | YES  |     | NULL    |       |
| OwnerUserId           | int(11)  | YES  |     | NULL    |       |
| OwnerDisplayName      | tinyblob | YES  |     | NULL    |       |
| LastEditorUserId      | int(11)  | YES  |     | NULL    |       |
| LastEditorDisplayName | tinyblob | YES  |     | NULL    |       |
| LastEditDate          | datetime | YES  |     | NULL    |       |
| LastActivityDate      | datetime | YES  |     | NULL    |       |
| Title                 | tinytext | YES  |     | NULL    |       |
| Tags                  | tinytext | YES  |     | NULL    |       |
| AnswerCount           | int(11)  | YES  |     | NULL    |       |
| CommentCount          | int(11)  | YES  |     | NULL    |       |
| FavoriteCount         | int(11)  | YES  |     | NULL    |       |
| ClosedDate            | datetime | YES  |     | NULL    |       |
| CommunityOwnedDate    | datetime | YES  |     | NULL    |       |
+-----------------------+----------+------+-----+---------+-------+

24. Load posts.xml data into the table:
mysql> LOAD XML LOCAL INFILE 'D:/FYP/stackoverflow.com-Posts/Posts.xml'
   	 -> INTO TABLE test_post;
Query OK, 37215528 rows affected, 1 warning (3 hours 46 min 43.14 sec)
Records: 37215528  Deleted: 0  Skipped: 0  Warnings: 1

25. Use python to connect to database ‘test’ and retrieve ‘non-null tags in the first 10 rows’:
>>> import MySQLdb as mdb
>>> import sys
>>> con = mdb.connect('localhost', 'root','l56530304T','test');
>>> cur = con.cursor();
>>> fetch_10_tags = "SELECT Tags from test_post WHERE Tags IS NOT NULL AND Id < 10;";
>>> cur.execute(fetch_10_tags);
3
>>> rows = cur.fetchall()
>>> for row in rows:
	print(row)
('<c#><winforms><type-conversion><decimal><opacity>',) #each line is a tuple 
('<html><css><css3><internet-explorer-7>',)
('<c#><.net><datetime>',)
>>> print(rows[1][0])
<html><css><css3><internet-explorer-7> #http://www.diveintopython.net/native_data_types/tuples.html


26. Fetch tags groups from DB, convert them from tuples to long list:
>>> fetch_all_tags = "SELECT Tags from test_post WHERE Tags IS NOT NULL";
>>> cur.execute(fetch_all_tags);
14458876
>>> tuples_of_tags = cur.fetchall()
>>> import re
>>> def tags_to_taglist(tags):
	tags_cleaned = re.sub("[<>]"," ", tags)
	tags_list = tags_cleaned.split()
	return tags_list

>>> tags_groups = [] #Initialize an empty list of tags_group

>>> for tuple in tuples_of_tags:
	tags_groups.append(tags_to_taglist(tuple[0]))
>>> print(len(tags_groups))
14458876
>>> print(tags_groups[0])
['c#', 'winforms', 'type-conversion', 'decimal', 'opacity']
>>> print(tags_groups[10000])
['visual-studio', 'resharper', 'visual-assist', 'viemu']

27. Train the model:
	>>> import logging
>>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
level=logging.INFO)
>>> num_features = 200 /100/300….
>>> min_word_count = 100
>>> num_workers = 2
>>> context = 2
>>> downsampling = 1e-3
>>> from gensim.models import word2vec
Warning (from warnings module):
  File "C:\Users\lint0011\AppData\Local\Programs\Python\Python36\lib\site-packages\gensim\utils.py", line 865
    warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
UserWarning: detected Windows; aliasing chunkize to chunkize_serial
2017-09-19 14:41:30,401 : INFO : 'pattern' package not found; tag filters are not available for English
>>> print ("Training model...")
Training model…
>>> model = word2vec.Word2Vec(tags_groups, workers=num_workers, \
			  size=num_features, min_count = min_word_count, \
window = context, sample = downsampling)
>>> model.init_sims(replace=True)
>>> model_name = "200features_100minwords_3context_for_tags_from_stackoverflow"
>>> model.save(model_name)

28. Finding the length of the dictionary
	>>> len(model.wv.vocab)
15749

29. Find number of times each vocab appears in the model: 
>>> vocab_obj = model.wv.vocab["python"]
>>> vocab_obj.count
806763
>>> vocab_obj = model.wv.vocab["c"]
>>> vocab_obj.count
257772

30. Find top N most similar words:
>>> model.most_similar("python", topn = 5)
[('python-2.7', 0.7518237829208374), ('python-3.x', 0.7127788066864014), ('perl', 0.51788729429245), ('c++', 0.515122652053833), ('python-3.4', 0.49681538343429565)]
31. Find word in dictionary according to index:
	>>> print(model.wv.index2word[4])
android

32. NaN : not a number

33. Use bottleneck to sort the smallest N numbers on the left/the largest top N numbers on the right
>>> import bottleneck as bn
>>> import numpy as np
>>> a = np.array([1,3,5,34,666,7])
>>> bn.partition(a, kth=2)
array([  1,   3,   5,  34, 666,   7])
>>> bn.partition(a, kth=3)
array([  1,   3,   5,   7, 666,  34])
34. Use heapsort in numpy built in:
>>> sorted = np.sort(a, kind = 'heapsort')
>>> sorted
array([  1,   3,   5,   7,  34, 666])

35. From what we have in the dictionary : model.wv.index2word[4] → andriod:
build (‘andriod’, 2345) # built as a tuple for each vocab and the # of times it appears
Assign them to a structured array : [ (‘andriod’, 2345), (‘python’, 806763),....]
>>> words_frequency_array = []
>>> for key, value in model.wv.vocab.items():
	count = value.count
	new_tuple = (key,count)
	words_frequency_array.append(new_tuple)	
>>> words_frequency_array[1000]
('reflector', 233)

Sort the array using numpy.sort
>>> a = np.array(values, dtype=dtype)       # create a structured array
>>> np.sort(a, order='height')                        
	array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
	       ('Lancelot', 1.8999999999999999, 38)],
 	     dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
>>> dtype = [('word','S50'),('frequency',int)]
>>> array_bf_sort = numpy.array(words_frequency_array, dtype = dtype)
>>> array_sorted = numpy.sort(array_bf_sort, order = 'frequency')
print(array_sorted)
[(b'.a',     100) (b'albumart',     100) (b'android-lru-cache',     100)
 ..., (b'c#', 1129784) (b'java', 1303971) (b'javascript', 1457906)]
>>> array_sorted
array([(b'.a',     100), (b'albumart',     100),
       (b'android-lru-cache',     100), ..., (b'c#', 1129784),
       (b'java', 1303971), (b'javascript', 1457906)],
      dtype=[('word', 'S50'), ('frequency', '<i4')])

36. Get top 100 frequenstly used tabs:
>>> len(array_sorted)
15749
>>> top_100_words = array_sorted[len(array_sorted)-100:] → reversed(...) #rank from biggest to smallest
>>> print(top_100_words)
[(b'google-chrome',   48793) (b'loops',   48853) (b'sockets',   48893)
 (b'pandas',   49350) (b'validation',   49963) (b'class',   50110)
 (b'sql-server-2008',   50475) (b'unit-testing',   51560)
 (b'symfony',   51896) (b'uitableview',   53274) (b'cordova',   53461)
 (b'google-maps',   53470) (b'reactjs',   53522) (b'file',   54626)
 (b'maven',   55072) (b'codeigniter',   55242)
 (b'ruby-on-rails-3',   55568) (b'api',   56070) (b'web-services',   56600)
 (b'rest',   57111) (b'sqlite',   57148) (b'perl',   57201)
 (b'shell',   57864) (b'.htaccess',   59089) (b'function',   59406)
 (b'qt',   60763) (b'linq',   65729) (b'hibernate',   65943)
 (b'excel-vba',   66306) (b'css3',   66882) (b'list',   67377)
 (b'swing',   67707) (b'angular',   67933) (b'laravel',   68795)
 (b'python-3.x',   68899) (b'entity-framework',   69589)
 (b'scala',   69910) (b'visual-studio',   71747) (b'performance',   72038)
 (b'postgresql',   73415) (b'python-2.7',   73808) (b'matlab',   73950)
 (b'apache',   74783) (b'winforms',   75961) (b'algorithm',   78154)
 (b'osx',   79726) (b'facebook',   80087) (b'image',   82749)
 (b'forms',   84129) (b'twitter-bootstrap',   84352) (b'vba',   86618)
 (b'mongodb',   86760) (b'oracle',   87848) (b'bash',   88526)
 (b'git',   89390) (b'multithreading',  100725) (b'html5',  106019)
 (b'eclipse',  108592) (b'windows',  112626) (b'vb.net',  112666)
 (b'spring',  116705) (b'string',  117093) (b'xcode',  117290)
 (b'wordpress',  119598) (b'excel',  128186) (b'database',  130766)
 (b'wpf',  133810) (b'django',  150763) (b'linux',  151867)
 (b'asp.net-mvc',  160311) (b'swift',  161272) (b'xml',  162576)
 (b'regex',  173692) (b'ajax',  175332) (b'ruby',  186001)
 (b'node.js',  188504) (b'r',  198930) (b'json',  208997)
 (b'sql-server',  209938) (b'iphone',  217987) (b'arrays',  236123)
 (b'angularjs',  240470) (b'.net',  257092) (b'c',  257772)
 (b'objective-c',  279158) (b'ruby-on-rails',  279812)
 (b'asp.net',  321621) (b'sql',  402719) (b'mysql',  480763)
 (b'css',  490186) (b'ios',  525190) (b'c++',  529922) (b'html',  683973)
 (b'python',  806763) (b'jquery',  864328) (b'android', 1021712)
 (b'php', 1114013) (b'c#', 1129784) (b'java', 1303971)
 (b'javascript', 1457906)]
	from biggest → smallest
37. Get top 10 most similar words for each words in top 100 list:
for pair in top_100_words:
	most_similar_10 = model.most_similar(str(pair[0],'utf-8'),topn = 10)
	print('{} is most similar with {} \n'.format(str(pair[0],'utf-8'), most_similar_10))
google-chrome is most similar with [('browser', 0.6782100200653076), ('safari', 0.6556639075279236), ('opera', 0.6083760261535645), ('cross-browser', 0.6074787378311157), ('firefox', 0.5976513624191284), ('internet-explorer', 0.579974889755249), ('firebug', 0.5780289173126221), ('microsoft-edge', 0.5409000515937805), ('mozilla', 0.5397083759307861), ('internet-explorer-9', 0.5363428592681885)] 

loops is most similar with [('for-loop', 0.6678728461265564), ('if-statement', 0.6657720804214478), ('while-loop', 0.6125882863998413), ('function', 0.5810701847076416), ('arrays', 0.5798569917678833), ('iteration', 0.5777946710586548), ('variables', 0.5738101005554199), ('foreach', 0.5673071146011353), ('nested-loops', 0.561725914478302), ('count', 0.5535820722579956)] 
….
…
ios is most similar with [('ios7', 0.5269947052001953), ('ios5', 0.513548731803894), ('ios4', 0.5051759481430054), ('ipad', 0.5037100911140442), ('ios6', 0.47771304845809937), ('cocoa-touch', 0.47586801648139954), ('cocoa', 0.469224214553833), ('iphone-sdk-3.0', 0.4679619073867798), ('xcode', 0.4679424464702606), ('ios8', 0.46788185834884644)] 
Group of ‘ios’
bf input into word2vec

c++ is most similar with [('c', 0.6885712146759033), ('visual-c++', 0.6629619598388672), ('pointers', 0.565581202507019), ('c++11', 0.5568311214447021), ('stl', 0.5466140508651733), ('gcc', 0.5286028385162354), ('assembly', 0.5235466957092285), ('matlab', 0.5205268859863281), ('python', 0.5151227116584778), ('c++-cli', 0.5008862018585205)] 

html is most similar with [('html5', 0.5867648124694824), ('xhtml', 0.5521109104156494), ('web', 0.5236678123474121), ('jquery', 0.5164123773574829), ('hyperlink', 0.5136469602584839), ('twitter-bootstrap', 0.5127034187316895), ('html-table', 0.499767929315567), ('website', 0.49311837553977966), ('javascript', 0.493013471364975), ('wordpress', 0.492295503616333)] 

python is most similar with [('python-2.7', 0.7518237829208374), ('python-3.x', 0.7127788066864014), ('perl', 0.5178873538970947), ('c++', 0.5151227116584778), ('python-3.4', 0.49681538343429565), ('bash', 0.4896581768989563), ('r', 0.48200365900993347), ('file', 0.4805043339729309), ('java', 0.4783886969089508), ('c', 0.46483245491981506)] 
‘-’ + numeric → eliminate
bf input into word2vec

jquery is most similar with [('jquery-plugins', 0.5401302576065063), ('angularjs', 0.5382768511772156), ('javascript', 0.532429575920105), ('html5', 0.5291421413421631), ('mootools', 0.5277490019798279), ('html', 0.5164123773574829), ('jquery-ui', 0.5066529512405396), ('jquery-mobile', 0.502914309501648), ('prototypejs', 0.5019865036010742), ('javascript-events', 0.4991496503353119)] 

android is most similar with [('android-activity', 0.5053296089172363), ('blackberry', 0.4806663393974304), ('java', 0.4790530204772949), ('mobile', 0.4678876996040344), ('android-studio', 0.45884326100349426), ('android-intent', 0.44837644696235657), ('javascript', 0.44363564252853394), ('android-asynctask', 0.42644190788269043), ('web', 0.4210183322429657), ('xcode', 0.4139876961708069)] 

php is most similar with [('mysqli', 0.5787276029586792), ('pdo', 0.5743492841720581), ('codeigniter', 0.572844386100769), ('zend-framework', 0.5569818019866943), ('ruby-on-rails', 0.5349618196487427), ('drupal', 0.5331737995147705), ('cakephp', 0.5329726934432983), ('perl', 0.5213042497634888), ('phpmyadmin', 0.5181658864021301), ('javascript', 0.5072661638259888)] 

c# is most similar with [('vb.net', 0.8491870164871216), ('c#-4.0', 0.7586356401443481), ('.net', 0.6869716644287109), ('.net-3.5', 0.5985285043716431), ('c#-3.0', 0.5589150190353394), ('.net-4.0', 0.551060676574707), ('silverlight', 0.546423077583313), ('asp.net', 0.538033664226532), ('wcf', 0.5290310978889465), ('visual-studio-2010', 0.5281020402908325)] 

java is most similar with [('scala', 0.5750092267990112), ('nullpointerexception', 0.5475189089775085), ('java-ee', 0.5392035245895386), ('eclipse', 0.5218921899795532), ('file', 0.498212605714798), ('exception', 0.48740410804748535), ('jsp', 0.487102210521698), ('servlets', 0.48273786902427673), ('android', 0.4790530204772949), ('python', 0.4783887267112732)] 

javascript is most similar with [('ajax', 0.5641095638275146), ('jquery', 0.532429575920105), ('angularjs', 0.524447500705719), ('dom', 0.5095491409301758), ('php', 0.5072661638259888), ('html5', 0.49794960021972656), ('html', 0.493013471364975), ('forms', 0.485580176115036), ('json', 0.4807314872741699), ('post', 0.47685718536376953)] 
only from the same category can be considered as similar!

38. Clean the database for similar tags: e.g. ‘python-2.7’ ‘python-3.x’...
	Use REGEXP to identify similar ending with ‘-’ + numeric value
	https://dev.mysql.com/doc/refman/5.7/en/regexp.html

	mysql> SELECT distinct Tags from test_post WHERE Tags REGEXP '.*-[0-9.]*';
	Result Summary: 3364618 rows in set (1 day 7 hours 39 min 42.29 sec)
	| <knife-solo>     as long as you have ‘-’ sign → fail to filter    
| <hash><embedded><lookup-tables>                                                                                           |
| <windows><winforms><c#-4.0><web>                                                                                          |
| <amazon-web-services><spring-boot><multipartform-data>                                                                    |
| <c#><ajax><asp.net-mvc><razor><asp.net-mvc-5>   
	
	 '.*-[0-9.]*’ → Improve:
mysql> SELECT '<windows><winforms><c#-4.0><web>'   REGEXP '.*-[0-9.]*$';
|                                                         0 |

mysql> SELECT '<windows><winforms><c#-4.0><web>'   REGEXP '.*-[0-9.]*>*';
|                                                    1 |

mysql> SELECT '<windows><winforms><c#-><web>'   REGEXP '.*-[0-9.]*>*';
|                                                       1 |

mysql> SELECT '<windows><winforms><c#-><web>'   REGEXP '.*-[0-9.]+>*';
|                                                       0 |

mysql> SELECT '<windows><winforms><c#-5.3><web>'   REGEXP '.*-[0-9.]+>*';
|                                                          1 |

Apply REGEXP '.*-[0-9.]+>*': 
mysql> SELECT distinct Tags from test_post WHERE Tags REGEXP '.*-[0-9.]+>*' LIMIT 10;
+---------------------------------------------------------------+
| Tags                                                          |
+---------------------------------------------------------------+
| <html><css><css3><internet-explorer-7>                        |
| <c#><linq><web-services><.net-3.5>                            |
| <actionscript-3><flex><bytearray>                             |
| <c#><linq><.net-3.5>                                          |
| <office-2007><file-type>                                      |
| <linq><.net-3.5>                                              |
| <flex><actionscript-3><air>                                   |
| <mysql><sql-server><csv><sql-server-2005><bcp>                |
| <mysql><sql-server><sql-server-2005>                          |
| <sql-server><sql-server-2005><deployment><release-management> |
+---------------------------------------------------------------+
10 rows in set (0.04 sec)

39. How to clean ‘-numeric’ value after identity them?
	https://stackoverflow.com/questions/986826/how-to-do-a-regular-expression-replace-in-mysql
Experiment with the first 10 rows in python as the one in step 38 above:
>>> fetch_10_tags = "SELECT Tags from test_post WHERE Tags REGEXP '.*-[0-9.]+>*' LIMIT 10;"
>>> cur.execute(fetch_10_tags)
10
>>> rows = cur.fetchall()
>>> for row in rows:
    print(row)  
('<html><css><css3><internet-explorer-7>',)
('<c#><linq><web-services><.net-3.5>',)
('<actionscript-3><flex><bytearray>',)
('<c#><linq><.net-3.5>',)
('<office-2007><file-type>',)
('<linq><.net-3.5>',)
('<flex><actionscript-3><air>',)
('<mysql><sql-server><csv><sql-server-2005><bcp>',)
('<mysql><sql-server><sql-server-2005>',)
('<sql-server><sql-server-2005><deployment><release-management>',)
>>> for row in rows:
	re.sub(r'-[0-9.]+>','>',row[0])	
'<html><css><css3><internet-explorer>'
'<c#><linq><web-services><.net>'
'<actionscript><flex><bytearray>'
'<c#><linq><.net>'
'<office><file-type>'
'<linq><.net>'
'<flex><actionscript><air>'
'<mysql><sql-server><csv><sql-server><bcp>'
'<mysql><sql-server><sql-server>'
'<sql-server><sql-server><deployment><release-management>'

40. Get the size of the table to determine the number of loops:
>>> con_get_size = mdb.connect('localhost', 'root','l56530304T','information_schema');
>>> cur = con_get_size.cursor()
>>> table_name = 'test_post'
>>> get_size = 'SELECT TABLE_ROWS FROM TABLES WHERE TABLE_NAME = \''+ table_name + '\''
>>> cur.execute(get_size)
1
>>> size = cur.fetchall()
>>> print(size[0][0])
26098009
>>> table_size = size[0][0]
>>> print(table_size)
26098009

Try out the first 10 rows:
>>> for id in range(10):
	fetch_one_row = 'SELECT Tags from test_post WHERE ID = '+str(id)+''
	cur.execute(fetch_one_row)
	rows = cur.fetchall()
	if rows and rows[0][0] is not None:
		cleaned = re.sub(r'-[0-9.]+>','>',rows[0][0])
		print(cleaned)
		update = 'UPDATE test_post SET Tags = \''+cleaned + '\' WHERE ID = '+str(id)
		cur.execute(update)

		
0
0
0
0
1
<c#><winforms><type-conversion><decimal><opacity>
0
0
1
<html><css><css3><internet-explorer>
0
1
0
1
<c#><.net><datetime>
0

41. Execute the cleaning process to get rid of ‘-number’:
>>> con = mdb.connect('localhost', 'root','l56530304T','test');
>>> cur = con.cursor()

>>> for id in range(table_size):
	for id in range(5006832,table_size):
	fetch_one_row = 'SELECT Tags from test_post WHERE ID = '+str(id)+''
	if(cur.execute(fetch_one_row)):
		rows = cur.fetchall()
		if rows[0][0] is not None:
			cleaned = re.sub(r'-[0-9.]+>','>',rows[0][0])
			print(cleaned + ' ' +str(id))
			update = 'UPDATE test_post SET Tags = \''+cleaned + '\' WHERE ID = '+str(id)
			cur.execute(update)

_mysql_exceptions.OperationalError: (1206, 'The total number of locks exceeds the lock table size')
Solution: in my_config.h add in ‘#define innodb_buffer_pool_size = 1G’

The last row being affected: <iphone><catransition> 5918078

41. mysql> select Tags,id from test_post ORDER BY id DESC LIMIT 1;
+------+----------+
| Tags | id       |
+------+----------+
| NULL | 45901678 |
+------+----------+
1 row in set (0.13 sec)
The problem is id is not consecutive, so have to find a way to loop through all rows w/o using id. 

42. Keming’s FYP for part 1 → Tag Categorization(from tagwiki)
	preprocess and retrieve -> tag with POS tagger -> extract category -> postprocess
	stackoverflow.com-Tags.7z 

42.5. Fail to install mysql-python package in Spyder :(
	


43. The method in step 41 does not guarantee that it is applicable to all cases and including all situation, therefore, we have to firstly extract all the tags that contain numeric values first to manually see what are the kinds, then design an inclusive way to filter away all number, version. 
	Firstly, we take out all tags with numbers inside:
Test the ‘SELECT’ sentence is correct:
		mysql> SELECT Tags, Id FROM test_post
    -> WHERE Id = 16
    -> AND Tags REGEXP '[0-9]';
+------------------------------------+----+
| Tags                                        | Id |
+------------------------------------+----+
| <c#><linq><web-services><.net-3.5> | 16 |
+------------------------------------+----+
1 row in set (0.00 sec)
	          2. Output all distinct tags into a file:
		Encountered Problem 1:
ERROR 1290 (HY000): The MySQL server is running with the --secure-file-priv option so it cannot execute this statement
Solution: 
Save to secure-file-priv location: C:\ProgramData\MySQL\MySQL Server 5.7\Uploads\
mysql> SELECT DISTINCT Tags FROM test_post
    -> WHERE Tags REGEXP '[0-9]'
    -> INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.7/Uploads/distinctTags.csv'
    -> FIELDS ENCLOSED BY '"';
#the above query was aborted, since this took too long and alternatively we can get a sense of all possible tags from ‘Tags.xml’
Encountered problem 2:
Regarding the above code, the runtime is extremely slow, can use indexing to speed up the process.
Database Indexing : https://www.tutorialspoint.com/dbms/dbms_indexing.htm
Database hashing: https://www.tutorialspoint.com/dbms/dbms_hashing.htm
Input Tags.xml into table test_tags, and output tags with TagNames that contain number:
mysql> CREATE TABLE test_tags(
    -> Id INT NOT NULL,
    -> TagName TINYTEXT NOT NULL,
    -> Count INT,
    -> ExcerptPostId INT,
    -> WikiPostId INT,
    -> PRIMARY KEY(Id)
    -> );
mysql> LOAD XML LOCAL INFILE 'D:/FYP/Tags.xml'
    -> INTO TABLE test_tags;
Query OK, 50000 rows affected (1.84 sec)
Records: 50000  Deleted: 0  Skipped: 0  Warnings: 0

mysql> SELECT DISTINCT TagName,Count FROM test_tags
    -> WHERE TagName REGEXP '[0-9]'
    -> INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.7/Uploads/TagsWithNumber.csv'
    -> FIELDS TERMINATED BY ','
    -> ENCLOSED BY '"';
Query OK, 4010 rows affected (0.06 sec)

	3. From ‘TagsWithNumber.csv’ classify them into types:
Sort them according the value → group the similar one together
Filter them with a minimal count of 1000
Pattern summary:
			Firstly, if encounter ‘-N(.Nx)’, then delete  ‘-N(.Nx)’ and all that follows
			
Otherwise, if it ends with ‘nameN(.Nx)’, then delete ‘N(.Nx)’								
			

			














	















