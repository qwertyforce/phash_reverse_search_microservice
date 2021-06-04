
import faiss
from os import listdir,remove
import numpy as np
import imagesize

import sqlite3
import io
conn = sqlite3.connect('phashes.db')
index=None
IMAGE_PATH="./../../../import/images"
      
def create_table():
	cursor = conn.cursor()
	query = '''
	    CREATE TABLE IF NOT EXISTS phashes(
	    	id INTEGER NOT NULL UNIQUE PRIMARY KEY, 
	    	phash BLOB NOT NULL
	    )
	'''
	cursor.execute(query)
	conn.commit()

def delete_descriptor_by_id(id):
	cursor = conn.cursor()
	query = '''DELETE FROM phashes WHERE id=(?)'''
	cursor.execute(query,(id,))
	conn.commit()

def get_all_ids():
    cursor = conn.cursor()
    query = '''SELECT id FROM phashes'''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    return list(map(lambda el:el[0],all_rows))

def get_all_data():
    cursor = conn.cursor()
    query = '''
    SELECT id, phash
    FROM phashes
    '''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    return list(map(lambda el:(el[0],convert_array(el[1])),all_rows))

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def sync_db():
    ids_in_db=set(get_all_ids())
    file_names=listdir(IMAGE_PATH)
    for file_name in file_names:
        file_id=int(file_name[:file_name.index('.')])
        if file_id in ids_in_db:
            ids_in_db.remove(file_id)
    for id in ids_in_db:
        delete_descriptor_by_id(id)   #Fix this
        print(f"deleting {id}")
    print("db synced")

def phash_reverse_search(target_features,query_id):
    D, I = index.search(np.array([target_features]), 5)
    similar=[]
    for i in range(5):
        if int(I[0][i])!=query_id and D[0][i]<=32:
            similar.append(int(I[0][i]))
    return similar

file_id_to_file_name_map={}

def dedup():
    global index
    index = faiss.read_index_binary("trained.index")
    all_data=get_all_data()
    image_ids=np.array([np.int64(x[0]) for x in all_data])
    phashes=np.array([x[1] for x in all_data])
    print(phashes.shape)
    index.add_with_ids(phashes, image_ids)    
    print("Index is ready")
    file_names=listdir(IMAGE_PATH)
    for file_name in file_names:
        file_id=int(file_name[:file_name.index('.')])
        file_id_to_file_name_map[file_id]=file_name
    deleted=[]
    for x in all_data:
        if x[0] in deleted:
            continue
        res=phash_reverse_search(x[1],x[0])
        if len(res)!=0:
            print(f'duplicate {x[0]} - {res}')
            res.append(x[0])
            images_id_res=[]
            for img_id in res:
                width, height = imagesize.get(f'{IMAGE_PATH}/{file_id_to_file_name_map[img_id]}')
                images_id_res.append((img_id,width*height))
            images_id_res.sort(key=lambda x: x[1],reverse=True)
            for i in range(1,len(images_id_res)):
                img_id=images_id_res[i][0]
                print(f'deleting {file_id_to_file_name_map[img_id]}')
                index.remove_ids(np.int64([img_id]))
                deleted.append(img_id)
                delete_descriptor_by_id(images_id_res[i][0])
                remove(f'{IMAGE_PATH}/{file_id_to_file_name_map[img_id]}')
            # print(images_id_res)

create_table()
sync_db()
dedup()


   
