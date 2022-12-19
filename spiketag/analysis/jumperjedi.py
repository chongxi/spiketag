import pandas as pd
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import IPython.core.display as display
import scipy.stats as stats

def plot_hist2d(jedi, bmi_pos):
    fig, ax = plt.subplots(1, jedi.n_trials, figsize=(3*jedi.n_trials, 3))
    for k in range(jedi.n_trials):
        #     _goal_pos = np.unique(np.stack(df.goal_pos[df.trial_no==k].to_numpy()), axis=0).ravel()
        #     _bmi_pos = np.stack(df.bmi_pos[(df.trial_no==k) & df.hd_stillness[df.trial_no==k]].to_numpy())
        _goal_pos = jedi.get_goal_pos(k)
        _bmi_pos  = bmi_pos[jedi.df.trial_no==k]
        bmi_hist2d = np.histogram2d(_bmi_pos[:, 0], -_bmi_pos[:, 1],
                                    bins=np.linspace(-50, 50, 51))[0]
        bmi_hist2d = gaussian_filter(bmi_hist2d, sigma=2)
        goal_center = (_goal_pos[0], -_goal_pos[1])
        goal_region = plt.Circle(
            goal_center, 15, color='w', linewidth=3, fill=False)
        ax[k].pcolormesh(jedi.X, jedi.Y, bmi_hist2d.T,
                        cmap='hot', shading='gouraud')
        ax[k].add_patch(goal_region)
        ax[k].set_xlim(-50, 50)
        ax[k].set_ylim(-50, 50)
        ax[k].set_aspect('equal', 'box')
    return fig

def plot_two_boxes(dist1, dist2, dist1_name, dist2_name):
    dist_df = pd.DataFrame({dist1_name: dist1,
                            dist2_name: dist2})
    # Create box plot
    fig, ax = plt.subplots(figsize=(2, 4))
    sns.boxplot(data=dist_df, showfliers=False, ax=ax)
    plt.ylabel('distance to goal (cm)')
    plt.xticks(rotation=15)
    # Perform t-test
    p_value = stats.ttest_ind(dist1, dist2)[1]
    
    return fig, p_value


class JEDI(object):
    def __init__(self, file_name):
        '''
        load pandas file
        '''
        self.df = pd.read_pickle(file_name)
        bmi_dict = self.df.to_dict('list')
        self.bmi_var = {}
        for key_name in bmi_dict.keys():
            self.bmi_var[key_name] = np.stack(bmi_dict[key_name])
        self.X, self.Y = np.linspace(-50, 50, 50), np.linspace(-50, 50, 50)

    def get_time(self, k):
        return self.df.bmi_t[self.df.trial_no == k].to_numpy()

    def get_goal_pos(self, k):
        _goal_pos = np.unique(
            np.stack(self.df.goal_pos[self.df.trial_no == k].to_numpy()), axis=0).ravel()
        return _goal_pos

    def get_bmi_pos(self, k, only_hd_still=True, only_still=False):
        if only_hd_still or only_still:
            if only_hd_still:
                # & ~self.df.mua_burst[self.df.trial_no==k]
                _bmi_pos = np.stack(self.df.bmi_pos[(
                    self.df.trial_no == k) & self.df.hd_stillness[self.df.trial_no == k]].to_numpy())
            if only_still:
                _bmi_pos = np.stack(self.df.bmi_pos[(
                    self.df.trial_no == k) & self.df.stillness[self.df.trial_no == k]].to_numpy())
        else:
            _bmi_pos = np.stack(
                self.df.bmi_pos[(self.df.trial_no == k)].to_numpy())
        return _bmi_pos

    def get_replay_bmi_pos(self, k):
        _bmi_pos = np.stack(self.df.bmi_pos[(self.df.trial_no == k) & (
            self.df.mua_burst[self.df.trial_no == k] == True)].to_numpy())
        return _bmi_pos

    def get_ball_v(self, k):
        return np.stack(self.df.ball_v[self.df.trial_no == k].to_numpy())

    def get_stillness(self, k):
        return self.df.stillness[self.df.trial_no == k].to_numpy()

    def get_mua_burst(self, k):
        return self.df.mua_burst[self.df.trial_no == k].to_numpy()

    @property
    def bmi_t(self):
        return self.df.bmi_t.to_numpy()
    
    @property
    def jedi_start_time(self):
        return self.get_time(0)[0]

    @property
    def jedi_end_time(self):
        return self.get_time(self.df.trial_no.max())[-1]

    @property
    def n_trials(self):
        return self.df.trial_no.max() + 1

    @property
    def trial_times(self):
        trial_times = [[self.get_time(i)[0], self.get_time(i)[-1]] for i in range(self.n_trials)]
        return trial_times

    def __repr__(self):
        display(self.df)
        return ''

    def plot_2dhist(self, hd_stillness=True, mua_burst=False, goal_radius=15, exclude_frames=0):
        fig, ax = plt.subplots(1, self.n_trials, figsize=(3*self.n_trials, 3))
        for k in range(self.n_trials):
            _goal_pos = self.get_goal_pos(k)
            if mua_burst is False:
                _bmi_pos = self.get_bmi_pos(k, only_hd_still=hd_stillness)
            elif mua_burst is True:
                _bmi_pos = self.get_replay_bmi_pos(k)
            _bmi_t = self.get_time(k)
            _bmi_stillness_duration = len(_bmi_pos)/10
            _bmi_duration = _bmi_t[-1] - _bmi_t[0]
            if(hd_stillness is False):
                _bmi_stillness_duration = _bmi_duration
            if(_bmi_duration > 180):
                _bmi_duration = 180
            if(_bmi_stillness_duration > 180):
                _bmi_stillness_duration = 180

            _bmi_pos = _bmi_pos[exclude_frames:]
            bmi_hist2d = np.histogram2d(_bmi_pos[:, 0], -_bmi_pos[:, 1],
                                        bins=np.linspace(-50, 50, 51))[0]
            bmi_hist2d = gaussian_filter(bmi_hist2d, sigma=2)
            goal_center = (_goal_pos[0], -_goal_pos[1])
            goal_region = plt.Circle(
                goal_center, goal_radius, color='w', linewidth=2.5, fill=False, alpha=1)
            ax[k].pcolormesh(self.X, self.Y, bmi_hist2d.T,
                             cmap='coolwarm', shading='gouraud')
            ax[k].add_patch(goal_region)
            # draw a cross at x=0 with a length of 0.1
            ax[k].axhline(0, xmin=0.45, xmax=0.55, color='w')
            ax[k].axvline(0, ymin=0.45, ymax=0.55, color='w')

            ax[k].set_xlim(-50, 50)
            ax[k].set_ylim(-50, 50)
            ax[k].set_xticklabels([])
            ax[k].set_yticklabels([])
            ax[k].set_aspect('equal', 'box')
            ax[k].set_title('trial {}: {:.1f}/{:.1f} s'.format(k,
                                                               _bmi_stillness_duration, _bmi_duration), fontsize=18)
        return fig

    def shuffle_goal_test(self, hd_stillness=True, mua_burst=False, n_shuffles=30, goal_radius=15, shuffled_goals=None, trials_to_shuffle=None):
        '''
        The shuffle_goal_test function shuffles the goal locations for a given set of trials in order to estimate the 
        goal-directness of the JEDI BMI performance. The function takes several input arguments:

        - hd_stillness: a boolean indicating whether to use BMI frames where the head direction is still
        - mua_burst: a boolean indicating whether to use BMI frames where there is a MUA burst
        - n_shuffles: the number of times to shuffle the goals (0 means no shuffling, instead, the center is used)
        - goal_radius: the radius of the goals
        - shuffled_goals: instead of create shuffled goals, use the given shuffled goals
        '''

        import random

        total_goal_pos = []
        total_bmi_pos = []
        total_random_goal_pos = []
        if trials_to_shuffle is None:
            trials_to_shuffle = np.arange(self.n_trials)
        else:
            trials_to_shuffle = np.array(trials_to_shuffle)
        if n_shuffles > 0:
            # shuffle goal localtions (n times) for robust estimation
            for i in range(n_shuffles):
                shuffle_trials = trials_to_shuffle.copy()
                while np.abs(shuffle_trials - trials_to_shuffle).sum() < 4:
                    random.shuffle(shuffle_trials)
                if hd_stillness is True:
                    goal_pos = self.bmi_var['goal_pos'][(self.df.trial_no.isin(
                        trials_to_shuffle)) & (self.bmi_var['hd_stillness'])]
                    bmi_pos = self.bmi_var['bmi_pos'][(self.df.trial_no.isin(
                        trials_to_shuffle)) & (self.bmi_var['hd_stillness'])]
                    random_goal_pos = np.vstack([self.bmi_var['goal_pos'][(self.bmi_var['trial_no'] == i) & (
                        self.bmi_var['hd_stillness'])] for i in shuffle_trials])
                if mua_burst is True:
                    goal_pos = self.bmi_var['goal_pos'][(self.df.trial_no.isin(
                        trials_to_shuffle)) & (self.bmi_var['mua_burst'] > 0)]
                    bmi_pos = self.bmi_var['bmi_pos'][(self.df.trial_no.isin(
                        trials_to_shuffle)) & (self.bmi_var['mua_burst'] > 0)]
                    random_goal_pos = np.vstack([self.bmi_var['goal_pos'][(self.bmi_var['trial_no'] == i) & (
                        self.bmi_var['mua_burst'] > 0)] for i in shuffle_trials])
                elif hd_stillness is False and mua_burst is False:
                    goal_pos = self.bmi_var['goal_pos'][(
                        self.df.trial_no.isin(trials_to_shuffle))]
                    bmi_pos = self.bmi_var['bmi_pos'][(
                        self.df.trial_no.isin(trials_to_shuffle))]
                    random_goal_pos = np.vstack([self.bmi_var['goal_pos'][(
                        self.bmi_var['trial_no'] == i)] for i in shuffle_trials])

                total_goal_pos.append(goal_pos)
                total_bmi_pos.append(bmi_pos)
                total_random_goal_pos.append(random_goal_pos)

        elif n_shuffles == 0:  # shuffled goals are all at the center (0,0)
            if hd_stillness is True:
                goal_pos = self.bmi_var['goal_pos'][(self.df.trial_no.isin(
                    trials_to_shuffle)) & (self.bmi_var['hd_stillness'])]
                bmi_pos = self.bmi_var['bmi_pos'][(self.df.trial_no.isin(
                    trials_to_shuffle)) & (self.bmi_var['hd_stillness'])]
                random_goal_pos = np.zeros_like(bmi_pos)
            if mua_burst is True:
                goal_pos = self.bmi_var['goal_pos'][(self.df.trial_no.isin(
                    trials_to_shuffle)) & (self.bmi_var['mua_burst'] > 0)]
                bmi_pos = self.bmi_var['bmi_pos'][(self.df.trial_no.isin(
                    trials_to_shuffle)) & (self.bmi_var['mua_burst'] > 0)]
                random_goal_pos = np.zeros_like(bmi_pos)
            elif hd_stillness is False and mua_burst is False:
                goal_pos = self.bmi_var['goal_pos'][(
                    self.df.trial_no.isin(trials_to_shuffle))]
                bmi_pos = self.bmi_var['bmi_pos'][(
                    self.df.trial_no.isin(trials_to_shuffle))]
                random_goal_pos = np.zeros_like(bmi_pos)

            total_goal_pos.append(goal_pos)
            total_bmi_pos.append(bmi_pos)
            total_random_goal_pos.append(random_goal_pos)

        bmi_pos = np.vstack(total_bmi_pos)
        goal_pos = np.vstack(total_goal_pos)
        random_goal_pos = np.vstack(total_random_goal_pos)

        goal_dist = np.linalg.norm(bmi_pos - goal_pos, axis=1)
        shuffle_goal_dist = np.linalg.norm(bmi_pos - random_goal_pos, axis=1)

        dist1 = np.clip(goal_dist-goal_radius, 0, np.inf)
        dist2 = np.clip(shuffle_goal_dist-goal_radius, 0, np.inf)

        # goal_dist[goal_dist<=goal_radius] = goal_radius
        # shuffle_goal_dist[shuffle_goal_dist<=goal_radius] = goal_radius

        # dist1 = goal_dist-goal_radius
        # dist2 = shuffle_goal_dist-goal_radius

        fig, p = plot_two_boxes(
            dist1, dist2, dist1_name='to goals', dist2_name='to shuffled goals')
        return fig, p

