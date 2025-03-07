

def prompt_messages(self, SYSTEM_PROMPT,table_description):
    sqls= self.skill_library.get_sql(random_=True)
    self.messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"**Existing Queries**: {sqls } \n**Table Description**: {table_description}"},
        ]