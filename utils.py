import arcpy


def unique_values(table, field=None, fields=None, sql_ready=False):
	#returns list of unique values in a given field, in a table
	if fields is not None:
		#check for uniqueness across multple columns, return flattened strings
		with arcpy.da.SearchCursor(table, fields) as cursor:

			#uniques = sorted({row[0] for row in cursor})
			uniques = sorted({row for row in cursor})

			if sql_ready:
				#remove unicode thing and change to tuple
				return str(tuple(uniques)).replace("u", "")
			else:
				return uniques

	else:
		with arcpy.da.SearchCursor(table, field) as cursor:

			#uniques = sorted({row[0] for row in cursor})
			uniques = sorted({row[0] for row in cursor})

			if sql_ready:
				#remove unicode thing and change to tuple
				return str(tuple(uniques)).replace("u", "")
			else:
				return uniques

def random_alphanumeric(n=6):
	import random
	chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	return ''.join(random.choice(chars) for i in range(n))

def add_missing_fields(editSchemaTable, matchToTable):
	"""
	find what fields are not in the editSchemaTable that exist in the matchToTable
	and add them to the editSchemaTable. Useful before appending tables
	"""

	#get lists of field names
	editFieldsNames = [field.name for field in arcpy.ListFields(editSchemaTable)] #listFieldNames(editSchemaTable) #arcpy.ListFields(editSchemaTable)
	matchFieldNames = [field.name for field in arcpy.ListFields(matchToTable)]

	#add necessary fields
	addFieldsList = []
	for fieldname in matchFieldNames:
		#create list of field names to be added
		if not fieldname.upper() in [s.upper() for s in editFieldsNames]:
			addFieldsList.append(fieldname)
			#print "add: " + fieldname

	for field in arcpy.ListFields(matchToTable):
		#print (field.name + " " + field.type.upper())
		if field.name.upper() in [s.upper() for s in addFieldsList]:
			#print ("adding " + field.name + " " + field.type.upper())
			arcpy.AddField_management(in_table = editSchemaTable, field_name = field.name, field_type = field.type.upper(), field_length = field.length)
