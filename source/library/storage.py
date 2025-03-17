


import os
import json
import random
from typing import List
import faiss
from source.library.utils import load_sentence
import numpy as np


class SQLLibrary:
    def __init__(self, data_args, model_args):
        self.data_args=data_args
        library_path = data_args.library_path
        # Load from disk or init empty
        if os.path.exists(library_path):
            with open(library_path, "r") as f:
                self.storage = json.load(f)
                print("There is already",len(self.storage),"Sql queries")
        else:
            self.storage = {}

        self.library_path = library_path

        self.model = load_sentence(model_args.sentence_model_name_or_path, model_args.hf_tokens)
        dim = self.model.get_sentence_embedding_dimension()#1024  #  the vector dimension 
        self.vect_index = faiss.IndexFlatIP(dim)# ini
        if len(self.storage)>0:
            self.vect_index.add(np.array([self.storage[str(i)]['embedding'] for i in range(len(self.storage))]
                                         )
                                )

        self.selected_index=[] # list of random selected index of sql skills 
        self.selected_ret_index = [] # list of retrieved selected index of sql skills ]
        
        
    def __repr__(self):
        return self.storage

    def __len__(self):
        return len(self.storage)


    def save(self):
        with open(self.library_path, "w") as f:
            json.dump(self.storage, f, indent=2)
    
    def get_sql(self,random_=True,num_q=5):
        n_skill=len(self.storage)
        if n_skill==0:
            return ["SELECT * \nFROM [table];" ]
        if random_: 
            skills_shuffle= list(self.storage.keys())
            
            random.shuffle(skills_shuffle)
            self.selected_index = random.sample(skills_shuffle,k=min(num_q,n_skill
                                                                )
                                                ) 
            selected_sql = [self.storage[i]["sql"] for  i in  self.selected_index]
        else:
            n_skill-=1
            selected_sql = [ self.storage[str(i)] for i in range(n_skill,n_skill-num_q,-1) ]
            self.selected_index = list(range(n_skill,n_skill-num_q,-1) )
        
        return selected_sql

    def add_query(self, sql: str, python_func:str="None",sql_embd =None,save:bool=False)-> None:
        """
        Add a new skill to the library with minimal info (only SQL).
        """


        #embedding_vec = self.compute_embedding(sql)
        
        #print(f"Embedding vector shape {sql_embd.shape} : {embedding_vec}\n\n")
        self.storage[str(len(self.storage))] = {
            "sql" : sql,
            "embedding": sql_embd[0].tolist(),
            "python_func": python_func
        }
        self.vect_index.add(sql_embd)
       
        if save or len(self.storage)+1 % self.data_args.save_skills_at_n ==0: #add later to argg
            self.save()
    

    


    def get_sim_queries(self, embedding_vec: List[float], top_k: int = 10, throushold:int=.9,) -> List[str]:
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
        ret_sql = [ self.storage[str(i)]['sql'] for i in self.selected_ret_index  ] if len(self.selected_ret_index)>0 else []
        
        return ret_sql
    
    def get_sim_queries_prev(self, embedding_vec: List[float], top_k: int = 1, throushold:int=.6,) -> str:
        """
        Check prev query if it is too diffrent 
        """
        prev =  np.array(self.storage[str(len(self.storage)-1)]["embedding"])[None,:]
        sec_cond= (np.dot(embedding_vec, prev.T)/ (np.linalg.norm(embedding_vec) * np.linalg.norm(prev)))[0][0]
        if sec_cond <throushold: return self.storage[str(len(self.storage)-1)]["sql"]
        else: return None

    



    def compute_embedding(self,text: str) -> List[float]:
        """
        Compute embedding (e.g. using sentence-transformers or OpenAI embeddings).
        Return a vector as a list of floats.
        """
        # Example pseudocode:
        # from sentence_transformers import SentenceTransformer
        embedding = self.model.encode([text], convert_to_tensor=False
                        ,batch_size=32,show_progress_bar=False,normalize_embeddings=True)
        

        return np.array(embedding, dtype=np.float32)