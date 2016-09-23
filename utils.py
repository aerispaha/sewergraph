import arcpy

def matchSchemas(matchToTable, editSchemaTable):

	#find fields to remove from editSchemaTable (those in editSchemaTable but not in matchToTable)

	#get lists of field names
	hiddenFieldsNames = [] #listHiddenFields(matchToTable) #this was crashing
	matchFieldNames = [field.name for field in arcpy.ListFields(matchToTable)]
	editFieldsNames = [field.name for field in arcpy.ListFields(editSchemaTable)] #listFieldNames(editSchemaTable) #arcpy.ListFields(editSchemaTable)

	#remove subtypes
	#subtypeList = ["1", "2", "3", "4", "5", "6"]# these exist in the Dataconv

	#create list of fields to drop from the edit Schema table
	dropFieldsList = []
	for fieldname in editFieldsNames:
		#compare field names (convert temporarily to upper)
		if (not fieldname.upper() in [s.upper() for s in matchFieldNames]
			and not fieldname.upper() in [s.upper() for s in hiddenFieldsNames]):
			#print "drop: " + fieldname
			dropFieldsList.append(fieldname)

	#concatentate list and drop the fields
	dropFields = ";".join(dropFieldsList)

	arcpy.DeleteField_management(in_table=editSchemaTable, drop_field=dropFields)

	#add necessary fields
	addFieldsList = []
	for fieldname in matchFieldNames:
		#create list of field names to be added
		if not fieldname.upper() in [s.upper() for s in editFieldsNames]:
			addFieldsList.append(fieldname)
			print "add: " + fieldname

	for field in arcpy.ListFields(matchToTable):
		#print (field.name + " " + field.type.upper())
		if field.name.upper() in [s.upper() for s in addFieldsList]:
			print ("adding " + field.name + " " + field.type.upper())
			arcpy.AddField_management(in_table = editSchemaTable, field_name = field.name, field_type = field.type.upper(), field_length = field.length)
