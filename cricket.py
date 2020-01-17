

from datetime import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy
import pandas as pd
import plotly.express as px
from scipy import stats
from tqdm import tqdm
import yaml


def halfway_scatter(n, folder):
    file_ids = []
    halfway_scores = []
    final_scores = []
    years = []
    x = 0
    for file in os.listdir(folder):
        if file.endswith(".yaml"):
            x += 1
            if x > n:
                break
            with open(os.path.join(folder, file), 'r') as f:
                odi = yaml.safe_load(f)
            overs = odi['info']['overs']
            halfway = (overs / 2) + 0.1
            date = odi['info']['dates'][0]
            for innings in odi['innings']:
                score = 0
                halfway_score = 0
                inn_n = list(innings)[0]
                for ball in innings[inn_n]['deliveries']:
                    for b, details in ball.items():
                        if b == halfway:
                            halfway_score = score
                        score += details['runs']['total']

                file_ids.append(file.replace(".yaml",""))
                halfway_scores.append(halfway_score)
                final_scores.append(score)
                try:
                    years.append(date.year)
                except AttributeError as e:
                    years.append(date[0:4])

    d = {'file': file_ids,
         'year': years,
         'halfway_score': halfway_scores,
         'final_score': final_scores}

    df = pd.DataFrame(d)

    with open("summary.csv", "w+") as f:
        f.write(df.to_csv())

    plt.scatter(halfway_scores, final_scores, c=years, cmap='viridis')
    plt.colorbar()
    plt.show()


def find_halfway_point():
    df = pd.read_csv('summary.csv')
    scores_int = []
    halfway_points = []
    totals = []
    half_totals = []
    total_overs = []
    for scores_str in df['over_end_scores']:
        scores_str_list = scores_str.strip('[]').split(',')
        scores = list(map(lambda s: None if s == " None" else int(s), scores_str_list))
        scores.insert(0, 0)
        scores_int.append(scores)
    for innings_raw in scores_int:
        innings = list(filter(None, innings_raw))
        total = innings[-1]
        half_total = total // 2
        halfway = next(i for i, v in enumerate(innings) if v > half_total)
        halfway_points.append(halfway)
        totals.append(total)
        half_totals.append(half_total)
        total_overs.append(len(innings))

        if halfway == 40:
            print(total, len(innings))

    df['total'] = totals
    df['halfway_point'] = halfway_points
    df['total_overs'] = total_overs

    df = df.loc[df['total_overs'] >= 25]

    for i in range(10):
        test_pop = df['halfway_point'].sample(frac=0.05).reset_index()
        # print(test_pop)

        w, p = stats.shapiro(test_pop['halfway_point'])
        avg = numpy.mean(test_pop['halfway_point'])
        sd = numpy.std(test_pop['halfway_point'])

        print(f"** Test {i+1} **\n"
              f"Population: {len(test_pop)}\n"
              f"w-score: {w:.3f}\n"
              f"p-value: {p:.3f}\n"
              f"Mean: {avg:.1f}\n"
              f"Standard deviation: {sd:.1f}")
        print("-" * 30)

        # df_freq = test_pop.groupby('halfway_point').count().reset_index('halfway_point')
        # plt.bar(df_freq['halfway_point'], df_freq['index'])
        # plt.show()


def fix_dates():
    folder = 'odis'
    for file in os.listdir(folder):
        if file.endswith(".yaml"):
            with open(os.path.join(folder, file), 'r') as f:
                odi = yaml.safe_load(f)
                try:
                    date = odi['info']['dates'][0]
                    if isinstance(date, str):
                        odi['info']['dates'][0] = datetime.strptime(date, "%Y-%m-%d").date()
                    with open(os.path.join(folder, "fixed", file), 'w') as write_file:
                        yaml.dump(odi, write_file, default_flow_style=False)
                except TypeError as e:
                    print(file)


def over_scores(n, folder):
    file_ids = []
    over_end_scores = []
    dates = []
    x = 0
    for file in os.listdir(folder):
        if file.endswith(".yaml"):
            x += 1
            if x > n:
                break
            with open(os.path.join(folder, file), 'r') as f:
                odi = yaml.safe_load(f)
            overs = odi['info']['overs']
            halfway = (overs / 2) + 0.1
            date = odi['info']['dates'][0]
            for innings in odi['innings']:
                score = 0
                inn_n = list(innings)[0]
                over_end_score_list = []
                current_over = 0
                for ball in innings[inn_n]['deliveries']:
                    for b, details in ball.items():
                        if b//1 != current_over:
                            over_end_score_list.append(score)
                            current_over = b//1
                        score += details['runs']['total']
                over_end_score_list.append(score)
                file_ids.append(file.replace(".yaml",""))
                dates.append(date)
                overs_remaining = overs - len(over_end_score_list)
                if overs_remaining > 0:
                    over_end_score_list.extend([None]*overs_remaining)
                over_end_scores.append(over_end_score_list)

    d = {'file': file_ids,
         'date': dates,
         'over_end_scores': over_end_scores,
         }

    df = pd.DataFrame(d)

    with open("summary.csv", "w+") as f:
        f.write(df.to_csv())


def plot_from_csv():
    df = pd.read_csv("summary.csv")

    plt.scatter(df['halfway_score'],
                df['final_score'],
                c=df['years'],
                cmap='viridis')
    plt.colorbar()
    plt.show()


def plot_over_scores():
    df = pd.read_csv('summary.csv')
    scores_int = []
    for scores_str in df['over_end_scores']:
        scores_str_list = scores_str.strip('[]').split(',')
        scores = list(map(lambda s: None if s == " None" else int(s), scores_str_list))
        scores.insert(0, 0)
        scores_int.append(scores)
    for s in scores_int:
        plt.plot(s)
    plt.show()


def conversion_rate(from_csv=True):
    if from_csv:
        df = pd.read_csv('tests/tests.csv')
    else:
        matches = {}
        files = list(Path('tests').glob('**/*.yaml'))
        for file in tqdm(files):
            match = yaml.safe_load(file.read_text())
            for innings in match['innings']:
                players = {}
                for key, value in innings.items():
                    if key[-7:] == 'innings':
                        for ball in value['deliveries']:
                            for b, details in ball.items():
                                try:
                                    players[details['batsman']] += details.get('runs', 0).get('batsman', 0)
                                except KeyError:
                                    players[details['batsman']] = details.get('runs', 0).get('batsman', 0)

                    matches[file.stem] = {key: players}

        df = pd.DataFrame.from_dict({(i, j, k): matches[i][j][k] for i in matches.keys()
                                     for j in matches[i].keys()
                                     for k in matches[i][j].keys()},
                                    orient='index').reset_index()
        df[['match', 'innings', 'batsman']] = pd.DataFrame(df['index'].tolist(), index=df.index)
        df = df.drop('index', axis=1)
        df = df.rename(columns={0: 'runs'})
        df.to_csv('tests/tests.csv')

    df = df[['batsman', 'runs']]
    all_innings = df.groupby('batsman').count()
    filtered = df[df['runs'] >= 10].groupby('batsman').count()

    conversion = all_innings.merge(filtered, on='batsman', how='left', suffixes=('_all', '_filtered'))
    conversion['ratio'] = conversion['runs_filtered'] / conversion['runs_all']
    conversion = conversion.fillna(0)
    conversion = conversion.sort_values('ratio', ascending=False)

    print(conversion.loc['JL Denly'])


if __name__ == "__main__":
    # halfway_scatter(10000, 'odis')
    # over_scores(10000, 'odis')
    # find_halfway_point()
    conversion_rate(from_csv=True)
