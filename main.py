import requests
import csv
import json
import datetime


def collect_data(apikey):
    file = open('movies-250.txt', 'r', encoding='utf-8')
    dataset = open(f'movies-{datetime.date.today()}.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(dataset)
    writer.writerow(['Title', 'Year', 'Rated', 'Genre', 'Plot'])
    collect = []
    cannot_collect = []
    for line in file:
        title = line.rstrip('\n')
        r = requests.get(f'http://www.omdbapi.com/?apikey={apikey}&t={title}&plot=full')
        obj = json.loads(r.text)
        if 'Error' in obj:
            cannot_collect.append(title)
            continue
        else:
            collect.append(title)
        writer.writerow([
            obj['Title'],
            obj['Year'],
            obj['Rated'],
            str(obj['Genre']).replace(', ', ';'),
            str(obj['Plot']).replace('"', "'")
        ])
    return {
        'retrieve': collect,
        'error': cannot_collect
    }

if __name__ == '__main__':
    obj = collect_data('')
    print(obj['error'])
