import arcpy
import os
#arcpy.env.overwriteOutput = True
def trace(gn, flags, flag_id, traced_features, out_gn, trace_task='TRACE_UPSTREAM'):

    """

    gn = path to Geometric Network
    flags = path to feature class (WITHIN the gn) with features used as flags
    flag_id = OBJECTID of flag element on which to base the trace
    traced_features = feature class within the gn to be traced upon
        Note: all feature classes within the gn are traced; this layer
        will be copied within the scope of this function
    out_gn = the outputed layer reference to the original gn (this is weird)
    trace_task = trace task completed
        options: {
            'TRACE_UPSTREAM' (default),
            'TRACE_DOWNSTREAM '
            }

    USAGE:
    pathToGdb =  r"C:\Data\Code\trace\test.gdb"
    geoNetwork = os.path.join(pathToGdb,'Hydro','Hydro_Net')
    damsFC = os.path.join(pathToGdb,'Hydro','DamPoints')
    gagesFC = os.path.join(pathToGdb,'Hydro','GagePoints')

    #run the goods
    trace(
        gn = geoNetwork,
        flags = 'DamPoints',
        flag_id = 512,
        traced_features = 'Flowline',
        out_gn = 'dda',
        trace_task = 'TRACE_DOWNSTREAM'
        )

    """

    arcpy.env.workspace = r'in_memory' #Group Layer resides in in_memory

    #FIRST SELECT THE INDIVIDUAL FLAG ELEMENT
    where = "OBJECTID = {}".format(flag_id)
    arcpy.SelectLayerByAttribute_management(flags, "NEW_SELECTION", where)

    #forget the map reference, and go straight to its source
    # downstreamUsers = []

    arcpy.TraceGeometricNetwork_management(gn, out_gn, flags, trace_task)
    trace_layer = 'trace_{}_{}'.format(flag_id, trace_task)
    arcpy.CopyFeatures_management(os.path.join(out_gn, traced_features), trace_layer)


    #collect the ids of elements traced
    traced_cursor = arcpy.SearchCursor(trace_layer)
    traced_ids = []
    for row in traced_cursor:
        traced_ids.append(row.getValue("OBJECTID"))

    del traced_cursor
    # for layer in arcpy.mapping.Layer(outNet): #referencing Group Layer in_memory
    #     for x in arcpy.mapping.ListLayers(layer): #ListLayers works in_memory too, to list layers inside of group layer
    #         if x.name == userLayer:
    #             print x.name, ' - lyrrrr'
    #             rows = arcpy.da.SearchCursor(userLayer, 'OBJECTID') #Better to use the newer search cursor with a specified field
    #             for row in rows:
    #                 downstreamUsers.append(row[0]) #cursors make tuples, and you want elements in the 0th position of the tuple

    arcpy.Delete_management(out_gn)
    #print "downstream user IDs", downstreamUsers
