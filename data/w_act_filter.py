"""
Filter BPIC 2012 CSV to only W_ activities (work items) with resources
Matches the repository's preprocessing but keeps resource information
"""

import csv
from collections import defaultdict

def filter_to_w_activities(input_csv, output_csv):
    """
    Filter to only W_ activities and map to integer IDs with resources
    """
    
    # Read all events
    events_by_case = defaultdict(list)
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            activity = row['Activity']
            
            # Only keep W_ activities (work items - manually executed)
            if activity.startswith('W_'):
                events_by_case[row['CaseID']].append({
                    'activity': activity,
                    'timestamp': row['CompleteTimestamp'],
                    'resource': row['Resource']
                })
    
    # Get unique W_ activities and create mapping
    all_activities = set()
    for events in events_by_case.values():
        for event in events:
            all_activities.add(event['activity'])
    
    # Sort for consistent mapping
    sorted_activities = sorted(all_activities)
    activity_to_id = {act: idx + 1 for idx, act in enumerate(sorted_activities)}
    
    print(f'Found {len(sorted_activities)} W_ activities:')
    for act, aid in sorted(activity_to_id.items(), key=lambda x: x[1]):
        print(f'  {aid}: {act}')
    
    # Write filtered data
    total_events = 0
    total_cases = 0
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['CaseID', 'ActivityID', 'CompleteTimestamp', 'Resource'])
        
        for case_id in sorted(events_by_case.keys()):
            events = events_by_case[case_id]
            
            # Sort events by timestamp
            events.sort(key=lambda x: x['timestamp'])
            
            for event in events:
                activity_id = activity_to_id[event['activity']]
                writer.writerow([
                    case_id,
                    activity_id,
                    event['timestamp'],
                    event['resource']
                ])
                total_events += 1
            
            total_cases += 1
    
    print(f'\nFiltered dataset:')
    print(f'  Total cases: {total_cases}')
    print(f'  Total events: {total_events}')
    print(f'  Average events per case: {total_events/total_cases:.2f}')
    print(f'  Output: {output_csv}')
    
    return activity_to_id

if __name__ == '__main__':
    input_csv = 'bpic2012_with_resources.csv'
    output_csv = 'bpic2012_w_resources.csv'
    
    print('Filtering BPIC 2012 to W_ activities with resources...\n')
    activity_mapping = filter_to_w_activities(input_csv, output_csv)
    
    print('\nSample output:')
    with open(output_csv, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 15:
                print(line.strip())
            else:
                break
    
    print('\n✅ Ready to use for training with resources!')