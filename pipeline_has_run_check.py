import sqlite3

class PipelineHasRunCheck:
	def __init__(self, db_path="db_file.db"):
		self.db_path = db_path
		with sqlite3.connect(db_path) as conn:
			cursor = conn.cursor()
			cursor.execute('''
				CREATE TABLE IF NOT EXISTS pipeline_has_run (
					id INTEGER PRIMARY KEY CHECK (id = 1),
					has_run BOOLEAN NOT NULL DEFAULT 0 CHECK (has_run IN (0, 1))
				)
			''') # the check constraint on the id ensures theres only ever 1 row existing on this table
			
			cursor.execute("INSERT OR IGNORE INTO pipeline_has_run (id, has_run) VALUES (1,FALSE)")
			conn.commit()
	
	
	def mark_as_run(self):
		with sqlite3.connect(self.db_path) as conn:
			cursor = conn.cursor()
			cursor.execute("""INSERT OR REPLACE INTO pipeline_has_run (id, has_run) 
VALUES (1, TRUE)""")
			conn.commit()
	
	
	def check(self):
		with sqlite3.connect(self.db_path) as conn:
			cursor = conn.cursor()
			cursor.execute("SELECT has_run FROM pipeline_has_run WHERE id = 1")
			return bool(cursor.fetchone()[0])