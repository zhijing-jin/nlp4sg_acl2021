class Constants:
    file_paper_list_long = 'data/acl20_long.txt'
    file_paper_list_short = 'data/acl20_short.txt'
    file_paper_list_long_csv = 'data/acl20_long.csv'
    file_paper_list_short_csv = 'data/acl20_short.csv'
    file_paper_list_long_xlxs = 'data/acl20_long.xlsx'
    img = 'data/visualization/acl20.png'
    img_dir = 'data/visualization/'

    @staticmethod
    def read_csv(file, verbose=False):
        import csv

        with open(file) as f:
            dialect = csv.Sniffer().sniff(f.readline(), delimiters=";,")
            f.seek(0)
            reader = csv.DictReader(f, delimiter=dialect.delimiter)
            lines = [i for i in reader if any(x.strip() for x in i)]
        if verbose:
            from efficiency.log import show_var
            show_var(['file', 'len(lines)', 'lines[:2]'])
        return lines

    @staticmethod
    def write_to_csv(header, rows, file):
        import csv
        to_write = [header] + rows
        print('[Info] Writing {} rows to {}'.format(len(to_write), file))
        with open(file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(to_write)


class Stats:
    def count(self):
        from efficiency.log import fread, show_var
        from efficiency.function import random_sample
        content = fread(C.file_paper_list_short, if_strip=True)
        for i in range(0, len(content), 3):
            try:
                content[i + 1]
            except:
                show_var(['content[i]'])
                import pdb;
                pdb.set_trace()
        papers = [(content[i], content[i + 1]) for i in
                  range(0, len(content), 3) if content[i]]
        show_var(['len(papers)', 'papers[0]', 'papers[-1]'])

        papers = sorted(papers)
        selected = random_sample(papers, 20)
        show_var(['len(selected)', 'selected[0]', 'selected[-1]', 'selected'])

        rand = random_sample(papers)

        C.write_to_csv(['paper_name', 'author'], rand,
                       C.file_paper_list_short_csv)

    def set_heads(self, header):
        self.head_stage = [i for i in header if i.startswith('Stage')][0]
        self.head_track = [i for i in header if i.startswith('track')][0]
        self.head_good = [i for i in header if i.startswith('Social Good')][0]
        self.head_un_goal = [i for i in header if i.startswith('UN 6 goals')][0]
        self.head_country = [i for i in header if i.startswith('country')][0]
        self.head_aff = [i for i in header if i.startswith('affiliaton')][0]
        self.head_if_company = [i for i in header if i.startswith(
            'if first author is supported by companies')][0]
        self.head_bin_company = 'if company (binary)'
        self.header = header + [self.head_bin_company]

    def read(self):
        from efficiency.function import random_sample

        file = C.file_paper_list_long_csv
        contents = C.read_csv(file)

        header = list(contents[0].keys())[1:-1]
        self.set_heads(header)

        def _parse_cell(cell, return_one=False):
            items = [i.strip() for i in cell.split(';')]
            items = [i for i in items if i]
            if return_one:
                return random_sample(items, 1)
            return items

        cleaned = []
        for row in contents:
            if not row[self.head_stage]:
                continue
            item = {h: _parse_cell(row[h], return_one=h == self.head_stage)
                    for h in header}
            item[self.head_bin_company] = [
                'Academia' if row[self.head_if_company] == 'N' else
                'Industry']
            if row[self.head_if_company] == 'N':
                item[self.head_if_company]= ['Academia']
            cleaned.append(item)
        return cleaned

    def make_all_plots(self, contents, header,
                       fig_templ='data/visualization/acl20_head{}.png', stacked_plot=True):
        from efficiency.function import flatten_list
        from collections import Counter

        P = Plotter()

        self.print_stage_n_track(contents)

        class2cnt = {}
        total_num = {}
        for head_i, head in enumerate(header):
            to_plot = [i[head] for i in contents if i[head]]
            to_plot = flatten_list(to_plot)
            class2cnt[head] = Counter(to_plot)
            total_num[head] = len(to_plot)

        for head_i, head in enumerate(header):
            if head == self.head_stage:
                sorted_cnt = sorted(class2cnt[head].most_common())
            else:
                sorted_cnt = class2cnt[head].most_common()
            plt = P.plot_bar(sorted_cnt)
            fig_name = fig_templ.format(head_i)
            self.plot_add_meta_info(plt, head, total_num, fig_name)

            if stacked_plot:

                for head1 in [self.head_if_company, self.head_country,
                              self.head_aff, self.head_bin_company]:
                    if head1 == head:
                        continue
                    try:head_i1 = header.index(head1)
                    except:
                        from efficiency.log import show_var
                        show_var(['head1', 'header'])
                        import pdb;pdb.set_trace()
                    fig_name = C.img_dir + 'acl20_head{}n{}.png'.format(head_i, head_i1)
                    joint2cnt = self.get_joint_dist(contents, head, head1)
                    plt = P.stacked_bar(joint2cnt, head, class2cnt[head],
                                        class2cnt[head1], norm_by = total_num[head])

                    self.plot_add_meta_info(plt, head, total_num, fig_name)

    def plot_add_meta_info(self, plt, head, total_num, fig_name):
        plt.xlabel(
            '{} (# Relevant Papers: {})'.format(head, total_num[head]))
        if head == self.head_good:
            plt.ylabel('Percentage among Social Good Papers')
        else:
            plt.ylabel('Percentage of Papers')

        # plt.show()
        plt.tight_layout()
        plt.savefig(fig_name, dpi=150)
        plt.clf()

    def get_joint_dist(self, contents, head0, head1, norm_by_head0=False):
        from efficiency.function import flatten_list
        from collections import Counter

        stage_n_track = [(y, x) for i in contents for y in i[head0]
                         for x in i[head1]]
        joint2cnt = Counter(stage_n_track)
        # tracks = [i[header[1]] for i in cleaned]
        # tracks_cnt = tracks
        # for track in sorted(tracks):
        if norm_by_head0:
            to_plot = [i[head0] for i in contents]
            head0_to_cnt = Counter(flatten_list(to_plot))
            for i in joint2cnt:
                joint2cnt[i] /= head0_to_cnt[i[0]]
        return joint2cnt

    def print_stage_n_track(self, contents):
        from efficiency.log import show_var

        stage_n_track_cnt = self.get_joint_dist(
            contents, self.head_stage, self.head_track, norm_by_head0=True)

        stage_n_track_cnt_sorted = sorted(stage_n_track_cnt.most_common(),
                                          key=lambda i: (i[0][0], -i[-1]))
        show_var(['stage_n_track_cnt_sorted'])


class Plotter:
    def wrapper(self, plt, keys):
        from matplotlib.ticker import PercentFormatter

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        if len(keys) > 20:
            plt.xticks(rotation=90)
        elif len(keys) > 10:
            plt.xticks(rotation=70)
        return plt

    def plot_bar(self, sorted_cnt, max_cut_until=30):
        import numpy as np
        import matplotlib.pyplot as plt
        keys, quant = zip(*sorted_cnt)
        quant_sum = sum(quant)
        quant_norm = np.array(quant) / quant_sum

        plt.figure(dpi=150, tight_layout=True)
        plt.bar(keys[:max_cut_until], quant_norm[:max_cut_until])

        plt = self.wrapper(plt, keys)
        return plt

    def stacked_bar(self, joint2cnt, head0, head0_cnt, head1_cnt, norm_by=1, max_num_xlabel=30,
                    max_num_legend=9, width=0.35):
        import numpy as np
        import matplotlib.pyplot as plt

        head0s, _ = zip(*head0_cnt.most_common()[:max_num_xlabel])
        if head0.startswith('Stage'):
            head0s = sorted(head0s)
        head1s, _ = zip(*head1_cnt.most_common())
        head1n0 = [[joint2cnt[(i, head1)] for i in head0s]
                   for head1 in head1s]
        head1n0 = np.array(head1n0)

        if len(head1s) > max_num_legend:
            head1s = list(head1s[:max_num_legend]) + ['Others']
            head1_others_n0 = head1n0[max_num_legend + 1:].sum(0)
            head1n0[max_num_legend] = head1_others_n0
            head1n0 = head1n0[:max_num_legend+1]

        ind = np.arange(len(head0s))  # the x locations for the groups
        bars = []
        for head1_i, _ in enumerate(head1s):
            bottom = head1n0[head1_i + 1:].sum(0)
            bar = plt.bar(ind, head1n0[head1_i] / norm_by, bottom=bottom /norm_by)
            bars.append(bar)
        # bars = [
        #     plt.bar(ind, head1n0[head1_i], width, bottom=head1n0[-head1_i-1:].sum(0))
        #     for head1_i, head1 in enumerate(head1s)
        # ]

        plt.xticks(ind, head0s)
        plt.legend((i[0] for i in bars), head1s)

        plt = self.wrapper(plt, head0s)

        return plt


def main():
    try:
        import efficiency
    except:
        import os
        os.system('pip install efficiency')

    s = Stats()
    contents = s.read()
    s.make_all_plots(contents, s.header)

    contents_good = [i for i in contents if i[s.head_good]]
    s.make_all_plots(contents_good, [s.head_if_company],
                     fig_templ=C.img_dir + 'acl20_head2_by_company.png',stacked_plot=False)
    s.make_all_plots(contents_good, [s.head_country],
                     fig_templ=C.img_dir + 'acl20_head2_by_country.png',stacked_plot=False)
    # s.count()


if __name__ == '__main__':
    C = Constants()
    main()
