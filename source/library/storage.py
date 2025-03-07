


import os
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import time


class SQLLibrary:
    def __init__(self, library_path):

        self.library_path = library_path
        # Load from disk or init empty
        if os.path.exists(library_path):
            with open(library_path, "rb", encoding="utf-8") as f:
                self.storage = json.load(f)
        else:
            self.storage = {}

        dim = 1024  #  the vector dimension 
        if len(self.storage)==0:
            self.vect_index = faiss.IndexFlatIP(dim)# ini

        else :self.vect_index = faiss.IndexFlatIP(dim)# to be done 

        self.selected_index=[] # list of random selected index of sql skills 
        self.selected_ret_index = [] # list of retrieved selected index of sql skills ]
        
    def __repr__(self):
        return self.storage

    def __len__(self):
        return len(self.storage)


    def save(self):
        with open(self.library_path, "wb", encoding="utf-8") as f:
            json.dump(self.storage, f, indent=2)
    
    def get_sql(self,random_=True,num_q=5):
        n_skill=len(self.storage)
        if n_skill==0:
            return ["SELECT * \nFROM [table];" ]
        if random_: 
            self.selected_index = random.sample(list(self.storage.keys()),k=min(num_q,n_skill
                                                                )
                                                ) 
            selected_sql = [self.storage[i]["sql"] for  i in  self.selected_index]
        else:
            n_skill-=1
            selected_sql = [ self.storage[i] for i in range(n_skill,n_skill-num_q,-1) ]
            self.selected_index = list(range(n_skill,n_skill-num_q,-1) )
        
        return selected_sql

    def add_query(self, sql: str, embedding_vec: List[float],python_func:str=None,save:bool=False)-> None:
        """
        Add a new skill to the library with minimal info (only SQL).
        """
        self.storage[len(self.storage)] = {
            "sql" : sql,
            "embedding": embedding_vec,
            "python_func": python_func
        }
        self.vect_index.add(embedding_vec)
        if save or len(self.storage)%100==0:
            self.save()
    
    def get_queries(self, embedding_vec: List[float], top_k: int = 10, throushold:int=.9,) -> List[str]:
        """
        Return up to top_k skill names sorted by similarity (desc).
        """
        # If skill library is empty, return empty
        if not self.storage:
            return []
        
        self.selected_ret_index=[]
        # Compute similarity
        sims, ret_index = self.vect_index.search(embedding_vec, k=top_k)
        sims = sims[0]# because only one query
        self.selected_ret_index=ret_index[0]# because only one query
        self.selected_ret_index = [ self.selected_ret_index[i] for i in range(len(sims)
                                                  )  if (sims[i] > throushold )]
        ret_sql = [ self.storage[i]['sql'] for i in self.selected_ret_index  ] if len(self.selected_ret_index)>0 else []
        
        return ret_sql
    



