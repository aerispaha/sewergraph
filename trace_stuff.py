import arcpy
import os
from utils import matchSchemas

arcpy.env.overwriteOutput = True


def upstream_drainage(flag_id):
    wwgn = r'Database Connections\DataConv.sde\DataConv.GISAD.Waste Water Network\DataConv.GISAD.WasteNetwork'
    flags = 'DataConv.GISAD.wwManhole'
    traced_features = 'wwGravityMain'
    areas = r'P:\06_Tools\trace\maps\tracedata.gdb\upstream_drainage_areas'
    out_buffer = r'temp_buffer'

    up_sewers = trace(wwgn, flags, flag_id, traced_features, return_type='layer')

    # #select elements, then create a buffer around them
    # where = "FACILITYID IN {}".format(tuple([str(x) for x in upstream_ids]))
    # arcpy.MakeFeatureLayer_management(traced_features, 'traced_layer', where)
    # arcpy.SelectLayerByAttribute_management('traced_layer', "NEW_SELECTION")


    arcpy.Buffer_analysis(up_sewers, out_buffer,
                          buffer_distance_or_field="100 Feet",
                          line_side="FULL", line_end_type="ROUND",
                          dissolve_option="ALL",
                          dissolve_field="", method="PLANAR")

    #update the data to retain which FACILITYID this area belongs to
    #matchSchemas(areas, out_buffer)
    arcpy.AddField_management(in_table = out_buffer, field_name = 'FACILITYID', field_type = 'GUID')
    arcpy.AddField_management(in_table = out_buffer, field_name = 'Name', field_type = 'TEXT')
    area_cursor = arcpy.UpdateCursor(out_buffer)
    for row in area_cursor:
        row.setValue('Name', flag_id)
        print 'updating {}'.format(flag_id)
        area_cursor.updateRow(row)

    del area_cursor

    arcpy.Append_management(out_buffer, areas)
    print out_buffer
    #arcpy.Delete_management(out_buffer)

    return areas


def trace(gn, flags, flag_id, traced_features,
          trace_task='TRACE_UPSTREAM',
          disabled_feature='DataConv.GISAD.wwVentPipe',
          return_type='list'):

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
    disabled_feature = a feature in the gn that should not be traced

    USAGE:
    wwgn = os.path.join('Database Connections',
                        'DataConv.sde',
                        'DataConv.GISAD.Waste Water Network',
                        'DataConv.GISAD.WasteNetwork')
    #run the goods
    trace(
        gn = wwgn,
        flags = 'DataConv.GISAD.wwManhole',
        flag_id = 93483,
        traced_features = 'wwGravityMain', #note the lack of DataConv.GISAD
        trace_task = 'TRACE_DOWNSTREAM'
        )

    """

    #arcpy.env.workspace = os.path.dirname(gn)#r'in_memory' #Group Layer resides in in_memory
    #arcpy.env.workspace = r'in_memory' #Group Layer resides in in_memory
    gn_env = os.path.dirname(gn)
    flags = os.path.join(gn_env, flags)
    out_gn = r'in_memory\outGroup' #os.path.join('in_memory', out_gn)
    #FIRST SELECT THE INDIVIDUAL FLAG ELEMENT

    #arcpy needs a layer before a selection can be made
    where = "OBJECTID = {}".format(flag_id)
    print 'making feature layer: \n\t{}, \n\tid = {}'.format(flags, flag_id)
    arcpy.MakeFeatureLayer_management(flags, 'flags_layer', where)
    arcpy.SelectLayerByAttribute_management('flags_layer', "NEW_SELECTION")


    for row in arcpy.SearchCursor("flags_layer"):
        print row.getValue('OBJECTID')
    #forget the map reference, and go straight to its source
    # downstreamUsers = []

    print 'performing {} on {}'.format(trace_task, gn)
    arcpy.TraceGeometricNetwork_management(
            gn,
            out_gn,
            "flags_layer",
            trace_task,
            in_disable_from_trace = 'DataConv.GISAD.wwVentPipe'
        )

    # for layer in arcpy.mapping.Layer(out_gn): #referencing Group Layer in_memory
    #     print layer.name


    trace_layer = 'trace_tmp'#.format(flag_id, trace_task)
    copied_features = os.path.join(out_gn, traced_features)
    print 'copying {} into: {}'.format(copied_features, trace_layer)
    arcpy.CopyFeatures_management(copied_features, trace_layer)


    #collect the ids of elements traced
    traced_cursor = arcpy.SearchCursor(trace_layer)
    traced_ids = []
    for row in traced_cursor:
        traced_ids.append(row.getValue("FACILITYID"))

    del traced_cursor
    arcpy.Delete_management(out_gn)
    arcpy.Delete_management('flags_layer')

    if return_type == 'layer':
        return trace_layer

    else:
        arcpy.Delete_management(trace_layer)
        #print "downstream user IDs", downstreamUsers
        return traced_ids
