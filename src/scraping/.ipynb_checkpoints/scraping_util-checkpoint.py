import pandas as pd
import numpy as np
import praw
from psaw import PushshiftAPI
import re
import requests
import time

reddit = praw.Reddit(client_id="_o78cyJghqkD-w",
                     client_secret="TBi-N01zb-0weOaD2jEBKhYYYsQ",
                     user_agent="b")

api = PushshiftAPI(reddit)

def post_df(subreddit, id):
    """
    
    Takes a subreddit name and post id and returns a dataframe of post information.
    
    """
    submission = reddit.submission(id = id)
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

#Code adapted from https://www.reddit.com/r/pushshift/comments/bfc2m1/capping_at_1000_posts/

def scraper(subreddits, posts = 35000):
    """
    
    Returns a dataframe of posts. 
    
    For each subreddit, this function queries the pushshift api.
    IDs are then extracted and passed into the 'post_df' function
    using Prawn to extract the relevant post information.
    The default number of posts per subreddit is 35000. To avoid an 
    endless loop if the API stops accepting requests, exceptions are counted
    and the loop is broken if requests fail 1000 times.
    
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
                                print(master_df.shape[0])
                            else:
                                pass
                        else:
                            pass
                    else:
                        break
                last = int(s['created_utc'])

            except:
                #I haven't figured out a sleep time between requests, and at some point, the API will become unhappy with this many.
                #This counts how many requests are rejected, and at 1000 rejections, breaks the loop and prints the 'last' time and 
                #Subreddit it stopped at so that you can restart the function
                exceptions += 1
                if exceptions == 1000:
                    print(sub, last)
                    break
    return master_df