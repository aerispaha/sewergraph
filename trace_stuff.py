import arcpy
import os
from utils import add_missing_fields, random_alphanumeric, unique_values
import hhcalculations
from hhcalcs.hydraulics import pipe
from collections import namedtuple

arcpy.env.overwriteOutput = True

#DATA CONNECTIONS
wwgn = r'Database Connections\DataConv.sde\DataConv.GISAD.Waste Water Network\DataConv.GISAD.WasteNetwork'
flags = 'DataConv.GISAD.wwManhole'
traced_features = 'wwGravityMain'
#areas = r'P:\06_Tools\trace\maps\tracedata.gdb\upstream_drainage_areas'
area_roles = r'P:\06_Tools\trace\maps\tracedata.gdb\DrainageAreaRoles'
sheds = r'P:\06_Tools\trace\maps\tracedata.gdb\newsubsheds'
traced_sewers = r'P:\06_Tools\trace\maps\tracedata.gdb\traced_ww_gravmains'

#define a named tuple for clarity on silly arcpy da modules
piperowcols = ['DrainageArea', 'Capacity', 'Velocity', 'TravelTime_min', 'Slope',
                'shapelength', 'Diameter', 'Height', 'Width', 'PIPESHAPE',
                'UpStreamElevation', 'DownStreamElevation', 'OBJECTID']
PipeRow = namedtuple("PipeRow", piperowcols)

def upstream_drainage(flag_id):

    #temp variables
    out_intersect = r'P:\06_Tools\trace\maps\tracedata.gdb\temp_interect'
    out_dissolve = r'P:\06_Tools\trace\maps\tracedata.gdb\temp_dissolve'
    temp_trace_sewer = r'P:\06_Tools\trace\maps\tracedata.gdb\temp_trace_sewer'

    #id to associate upstream area to sewer
    study_id = random_alphanumeric(8)

    #TRACE UPSTREAM AND SELECT THE INTERSECTING DRAINAGEA SHEDS
    up_sewers = trace(wwgn, flags, flag_id, traced_features, return_type='layer')
    arcpy.MakeFeatureLayer_management(sheds, 'shedslyr')
    arcpy.SelectLayerByLocation_management('shedslyr', 'INTERSECT', up_sewers['layer'])

    #SUM THE AREA OF THE UPSTREAM SHEDS and update the shed roles table
    #to keep track of which sheds drain into each FACILITYID (flag_id)
    with arcpy.da.SearchCursor('shedslyr', ['OID@', 'SHAPE@AREA']) as sheds_cursor:
        roles_cols = ['Area_OBJECTID', 'Downstream_FACILITYID']
        with arcpy.da.InsertCursor(area_roles, roles_cols) as roles:
            upstream_area = 0
            for shedrow in sheds_cursor:
                #print shedrow #.getValue("OBJECTID")
                upstream_area += shedrow[1]

                #add to roles table (shed OID, Downstream_FACILITYID)
                #NEED TO PREVENT DUPLICATES - confirm that row with same
                #Area_OBJECTID and Downstream_FACILITYID don't exist already
                uniq_roles = unique_values(area_roles, fields=roles_cols)
                uniq_roles_strings = [''.join(x) for x in uniq_roles]
                role_pair = ''.join((str(shedrow[0]), str(flag_id)))
                #convert role pairs to strings for comparison
                if role_pair not in uniq_roles_strings:
                    print 'added uniq role_pair {}'.format(role_pair)
                    roles.insertRow((shedrow[0], flag_id))
                else:
                    print 'not added the duplicate role {}'.format((shedrow[0], flag_id))



    #return upstream_area / 43560.0


    # arcpy.CopyFeatures_management('thiessenlyr', out_intersect)
    # arcpy.Dissolve_management (out_intersect, out_dissolve)

    #update the data to retain which FACILITYID this area belongs to
    # add_missing_fields(out_dissolve, areas)
    # area_cursor = arcpy.UpdateCursor(out_dissolve)
    # area_info = {}
    # for row in area_cursor:
    #     row.setValue('Name', study_id)
    #     row.setValue('FACILITYID', up_sewers['first_feat'])
    #     print 'updating {}'.format(flag_id)
    #     area_info.update({flag_id:row.shape.area})
    #     area_cursor.updateRow(row)
    # del area_cursor
    # arcpy.Append_management(out_dissolve, areas, schema_type = 'NO_TEST')

    #update data in the traced sewers table, then append to master table
    add_missing_fields(up_sewers['layer'], traced_sewers)
    uniq_ids = unique_values(traced_sewers,'FACILITYID', sql_ready=True)

    where = "MH2_GUID = '{}' AND FACILITYID NOT IN {}".format(flag_id, uniq_ids) #grab all directly drainning sewers
    #print where
    piperowcols[5] = 'SHAPE@LENGTH' #rename the shapelength to actual arcpy crap
    sewer_add_count = 0
    with arcpy.da.UpdateCursor(up_sewers['layer'], piperowcols, where) as sewer_cursor:

        for row in sewer_cursor:
            sewer = pipe.Pipe(da_cursor = PipeRow(*row)) #PipeRow is a namedtuple

            print 'updating {}, capacity={}'.format(sewer, sewer.capacity)
            row[0] = upstream_area / 43560.0 #report in acres
            row[1] = sewer.capacity
            row[2] = sewer.velocity
            row[3] = sewer.travel_time
            # row.setValue('DrainageAreaID', study_id)
            # row.setValue('DrainageArea', area_info[flag_id])

            sewer_cursor.updateRow(row)
            sewer_add_count += 1

        #del sewer_cursor
        # sewer_cursor = arcpy.UpdateCursor(up_sewers['layer'], where_clause=where)
        # hhcalculations.runCalcs(sewer_cursor)

    if sewer_add_count > 0:
        print 'appending {} sewers segments'.format(sewer_add_count)
        arcpy.MakeFeatureLayer_management(up_sewers['layer'], 'temp_sewer_lyr', where)
        arcpy.CopyFeatures_management('temp_sewer_lyr', temp_trace_sewer)
        arcpy.Append_management(temp_trace_sewer, traced_sewers, schema_type = 'NO_TEST')
    else:
        print 'no new sewers to append'


    arcpy.Delete_management(out_dissolve)
    arcpy.Delete_management(out_intersect)
    arcpy.Delete_management(temp_trace_sewer)

    #return area_info


def trace(gn, flags, flag_id, traced_features,
          trace_task='TRACE_UPSTREAM',
          disabled_features='DataConv.GISAD.wwVentPipe',
          return_type='list', env = 'C:\Data\ArcGIS\Default.gdb'):

    """

    returning a list comes in order of trace from the starting node

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
    out_gn = os.path.join(env, 'out_gn') #r'in_memory\outGroup' #os.path.join('in_memory', out_gn)
    flag_guid = ''
    first_traced_feature_guid = ''
    #FIRST SELECT THE INDIVIDUAL FLAG ELEMENT

    #arcpy needs a layer before a selection can be made
    #where = "OBJECTID = {}".format(flag_id)
    where = "FACILITYID = '{}'".format(flag_id)
    #print 'making feature layer: \n\t{}, \n\tid = {}'.format(flags, flag_id)
    arcpy.MakeFeatureLayer_management(flags, 'flags_layer', where)
    arcpy.SelectLayerByAttribute_management('flags_layer', "NEW_SELECTION")


    for row in arcpy.SearchCursor("flags_layer"):
        flag_guid = row.getValue('FACILITYID')
        #print flag_guid
    #forget the map reference, and go straight to its source
    # downstreamUsers = []

    #print 'performing {} on {}'.format(trace_task, gn)
    arcpy.TraceGeometricNetwork_management(
            gn,
            out_gn,
            "flags_layer",
            trace_task,
            in_disable_from_trace = disabled_features #'DataConv.GISAD.wwVentPipe'
        )

    trace_layer = os.path.join(env, 'trace_tmp') #r'in_memory\trace_tmp'#.format(flag_id, trace_task)
    copied_features = os.path.join(out_gn, traced_features)
    #print 'copying {} into: {}'.format(copied_features, trace_layer)
    arcpy.CopyFeatures_management(copied_features, trace_layer)

    #collect the ids of elements traced
    traced_cursor = arcpy.SearchCursor(trace_layer)
    traced_ids = []
    for row in traced_cursor:
        traced_ids.append(row.getValue("FACILITYID"))
        if row.getValue('MH2_GUID') == flag_guid:
            first_traced_feature_guid = row.getValue("FACILITYID")

    del traced_cursor
    arcpy.Delete_management(out_gn)
    arcpy.Delete_management('flags_layer')

    if return_type == 'layer':
        #first_traced_feature_id = traced_ids[0]
        return {'layer':trace_layer, 'first_feat':first_traced_feature_guid}

    else:
        arcpy.Delete_management(trace_layer)
        #print "downstream user IDs", downstreamUsers
        return traced_ids
