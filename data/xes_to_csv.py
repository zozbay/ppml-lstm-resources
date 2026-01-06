"""
Convert BPIC 2012 XES file to CSV with resources
Extracts: CaseID, Activity, CompleteTimestamp, Resource
"""

import xml.etree.ElementTree as ET
import csv
from datetime import datetime

def parse_xes_to_csv(xes_file, output_csv):
    """
    Parse XES file and extract events with resources to CSV
    """
    
    # Parse XES file
    tree = ET.parse(xes_file)
    root = tree.getroot()
    
    # XES namespace
    ns = {'xes': 'http://www.xes-standard.org/'}
    
    # Open CSV for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CaseID', 'Activity', 'CompleteTimestamp', 'Resource'])
        
        event_count = 0
        case_count = 0
        
        # Iterate through traces (cases)
        for trace in root.findall('.//xes:trace', ns):
            case_count += 1
            case_id = None
            
            # Get case ID
            for attr in trace.findall('.//xes:string[@key="concept:name"]', ns):
                case_id = attr.get('value')
                break
            
            if not case_id:
                continue
                
            # Iterate through events in this trace
            for event in trace.findall('.//xes:event', ns):
                activity = None
                timestamp = None
                resource = None
                lifecycle = None
                
                # Extract event attributes
                for attr in event:
                    key = attr.get('key')
                    
                    if key == 'concept:name':
                        activity = attr.get('value')
                    elif key == 'time:timestamp':
                        timestamp = attr.get('value')
                    elif key == 'org:resource':
                        resource = attr.get('value')
                    elif key == 'lifecycle:transition':
                        lifecycle = attr.get('value')
                
                # Only include COMPLETE events (like the repository does)
                if lifecycle and lifecycle.upper() == 'COMPLETE':
                    if activity and timestamp:
                        # Use UNKNOWN if no resource specified
                        if not resource:
                            resource = 'UNKNOWN'
                        
                        # Format timestamp
                        try:
                            # Parse ISO format timestamp
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            formatted_time = timestamp
                        
                        writer.writerow([case_id, activity, formatted_time, resource])
                        event_count += 1
                        
                        if event_count % 10000 == 0:
                            print(f'Processed {event_count} events from {case_count} cases...')
        
        print(f'\nDone! Extracted {event_count} events from {case_count} cases')
        print(f'Output saved to: {output_csv}')

if __name__ == '__main__':
    # Input/output files
    xes_file = 'BPI_Challenge_2012.xes'  # Your downloaded XES file
    output_csv = 'bpic2012_with_resources.csv'
    
    print('Converting XES to CSV with resources...')
    parse_xes_to_csv(xes_file, output_csv)
    
    print('\nSample of output:')
    with open(output_csv, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 10:
                print(line.strip())
            else:
                break