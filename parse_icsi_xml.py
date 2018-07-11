# give as input name of the file and output directory

transcript_folder="/data/Public_Audio_Datasets/icsi_meeting/ICSI_Meeting_Transcripts_LDC2004T04/transcripts/"
output_ground_truth_folder="/data/Public_Audio_Datasets/icsi_meeting/icsi_ground_truth/"

import os,sys
import numpy as np
from bs4 import BeautifulSoup
all_files_in_transcript_folder=os.listdir(transcript_folder)

for file in all_files_in_transcript_folder:
    if file=="preambles.mrt":
        continue
    elif file=="readme":
        continue
    else:
        file_name=file
        print file_name

    # file_name="Bdb001.mrt";
    infile=open(transcript_folder+file_name,"r")
    contents=infile.read()
    #print(contents)

    # read the file into BeautifulSoup
    xml_file=BeautifulSoup(contents,'xml')
    #xml_file.prettify()

    # dictionary that stores name of wav files , with key participant ID from the prior set
    participant_name_audio_file_dict = {}

    # from xml file identify Particiant and Channel tag
    for item in xml_file.find_all('Participant'):
        #print item['Name'] , item['Channel']
        if(item.has_attr('Channel')):
            participant_name_audio_file_dict[item['Name']]=item['Channel'];


    # write to a text file (append to prexisting if it exists)
        # write to a text file (append to prexisting if it exists)
    file = open(output_ground_truth_folder+file_name.strip('.mrt')+"_participant_audio_file_list.txt",'w')
    for key in participant_name_audio_file_dict:
    # write to a text file (append to prexisting if it exists)
            print file_name.strip('.mrt'),key, participant_name_audio_file_dict[key]
            file.write(file_name.strip('.mrt') + "\t" + key + "\t" + participant_name_audio_file_dict[key])
            file.write('\n')
    file.close()

    # printing the list
    #for key in participant_name_audio_file_dict:
    #    print file_name.strip('.mrt'),key, participant_name_audio_file_dict[key]

    # open a file to write the table to
    file = open(output_ground_truth_folder+file_name.strip('.mrt')+"_all_audio_segments.txt",'w')

    # extract segments from the transcripts
    for item in xml_file.find_all('Segment'):
        # Identify person of interest regions for each participant
        if item.has_attr('Participant'):
                # Identify noise regions for person of interest
                if item.find('NonVocalSound'):
                        # only noise, no speech in the non vocal sound
                        if item.text.isspace():
                            print item['Participant'], item['StartTime'] , item['EndTime'], "N"
                            file.write(item['Participant']+ "\t" +item['StartTime'] + "\t" + item['EndTime']+ "\t" + "N")
                            file.write("\n")
                        # noise and speech in the non vocal sound
                        else:
                            print item['Participant'], item['StartTime'] , item['EndTime'], "NV"
                            file.write(item['Participant'] + "\t" + item['StartTime'] + "\t" + item['EndTime']+ "\t" + "NV")
                            file.write("\n")
                # identify vocal regions for person of interest
                else:
                        print item['Participant'], item['StartTime'] , item['EndTime'], "V"
                        file.write(item['Participant']+ "\t" + item['StartTime'] + "\t" + item['EndTime']+ "\t" + "V")
                        file.write("\n")


            # Identify noise regions that are not by any person of interest
            elif item.has_attr('CloseMic'):
                print "noise", item['StartTime'] , item['EndTime'], "N"
                file.write("noise" + "\t" + item['StartTime'] + "\t" + item['EndTime'] + "\t" + "N")
                file.write("\n")
        else:
                continue



    # close file
    file.close()
