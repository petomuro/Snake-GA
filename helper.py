import matplotlib.pyplot as plt
import csv
from datetime import datetime


class Helper:
    def __init__(self):
        self.n_games = []
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0
        self.time = str(datetime.now().strftime("%Y%m%d%H%M%S"))

    def write_result_to_list(self, n_games, new_score):
        self.n_games.append(n_games)
        self.plot_scores.append(new_score)
        self.total_score += new_score
        mean_score = self.total_score / n_games
        self.plot_mean_scores.append(mean_score)

    def write_result_to_csv(self):
        with open('results/results' + self.time + '.csv', mode='a', newline='') as results_file:
            results_writer = csv.writer(
                results_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i in range(len(self.plot_scores)):
                results_writer.writerow(
                    [self.plot_scores[i], self.n_games[i], self.plot_mean_scores[i]])

    def display_graph(self, n_games, new_score):
        self.plot_scores.append(new_score)
        self.total_score += new_score
        mean_score = self.total_score / n_games
        self.plot_mean_scores.append(mean_score)

        plt.ion()
        plt.gcf()
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(self.plot_scores)
        plt.plot(self.plot_mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(self.plot_scores)-1,
                 self.plot_scores[-1], str(self.plot_scores[-1]))
        plt.text(len(self.plot_mean_scores)-1,
                 self.plot_mean_scores[-1], str(self.plot_mean_scores[-1]))
        plt.savefig('Training.png')


if __name__ == "__main__":
    helper = Helper()

    with open('results/results20210520223339.csv') as results_file:
        csv_reader = csv.reader(results_file, delimiter=';')
        line_count = 0

        for row in csv_reader:
            line_count += 1
            helper.display_graph(int(row[1]), int(row[0]))
            print(f'Processed {line_count} lines.')
