import numpy as np
import pandas as pd
import praw
from psaw import PushshiftAPI
import requests
import time

reddit = praw.Reddit(client_id="your_id",
                     client_secret="your_secret",
                     user_agent="your_agent")

api = PushshiftAPI(reddit)

def post_df(subreddit, submission):
    """Returns a dataframe of Reddit post information.

    Utilizes the PRAW wrapper to format a Reddit post into a Pandas dataframe. Series
    are generated for the following: post creation date, title, flair, text, edited
    status, upvotes/downvotes, number of comments, if it was gilded, any awards
    received, and the subreddit it came from.

    Args:
        subreddit (str): Name of subreddit post is from. Used to label dataframe row.
        submission (obj): A PRAW instance of a single post. This is automatically generated
          in the `scraper` function. For more information on PRAW, see
          https://praw.readthedocs.io/en/latest/getting_started/quick_start.html.

    Returns:
        A Pandas dataframe with each post as 1 row.
    """
    post_df = pd.DataFrame(index = [0])

    post_df['created'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(submission.created))
    post_df['title'] = submission.title
    post_df['flair'] = submission.link_flair_text
    post_df['text'] = submission.selftext
    post_df['edited'] = submission.edited
    post_df['ups'] = submission.ups
    post_df['down'] = submission.downs
    post_df['num_comments'] = submission.num_comments
    post_df['gilded'] = submission.gilded
    post_df['awards'] = submission.total_awards_received
    post_df['sub'] = subreddit

    return post_df

def scraper(subreddits, posts = 200000):
    """Retrieves posts from subreddits.

    This function queries the Pushshift API to obtain Reddit post IDs.
    The PRAW wrapper used to format each post. Only non-sticked posts that
    have not been removed are queried. Code adapted from
    https://www.reddit.com/r/pushshift/comments/bfc2m1/capping_at_1000_posts/.

    Args:
        subreddits (list): List of subreddits to query from. Elements must
          match how they appear in the actual Reddit URL.
        posts (int): Number of posts to retrieve from each subreddit (i.e., each
          element in `subreddits`). Default is 35,000.

    Returns:
        A Pandas dataframe. For formatting of dataframe, see the `post_df`
        documentation.
    """
    master_df = pd.DataFrame()
    for sub in subreddits:
        ids = []
        last = ''
        url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={sub}&fields=id,stickied,user_removed,mod_removed,created_utc'
        exceptions = 0
        while len(ids) < posts:
            try:
                request = requests.get('{}&before={}'.format(url,last))
                json = request.json()
                for s in json['data']:
                    if len(ids) < posts:
                        if s['stickied'] == False and 'user_removed' not in s and 'mod_removed' not in s:
                            #Making sure post is > 1 word (some pass through the 'user_removed'/'mod_removed' filter above)
                            submission = reddit.submission(id = s['id'])
                            if len(submission.selftext.split()) > 1:
                                post = post_df(sub, s['id'])
                                master_df = master_df.append(post)
                                ids.append(s['id'])
                                time.sleep(1)
                            else:
                                pass
                        else:
                            pass
                    else:
                        break
                last = int(s['created_utc'])

            except:
                #I haven't figured out a sleep time between requests, and at some point, the API will become unhappy with this many.
                #This counts how many requests are rejected, and at 10 rejections, breaks the loop and prints the 'last' time and
                #Subreddit it stopped at so that you can restart the function
                exceptions += 1
                if exceptions == 10:
                    print(sub, last)
                    break
    return master_df
