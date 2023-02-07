# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 31/1/2023 9:54 pm
fetch tweets by id
"""
# standard
from __future__ import print_function
import getopt
import os
import sys

import tweepy

from common.setup import logging as Logger

# Generate your own at https://apps.twitter.com/app
CONSUMER_KEY = 'Consumer Key (API key)'
CONSUMER_SECRET = 'Consumer Secret (API Secret)'
OAUTH_TOKEN = 'Access Token'
OAUTH_TOKEN_SECRET = 'Access Token Secret'

# batch size depends on Twitter limit, 100 at this time
batch_size = 100

def get_tweet_id(line):
    '''
    Extracts and returns tweet ID from a line in the input.
    '''
    (tagid,_timestamp,_sandyflag) = line.split('\t')
    (_tag, _search, tweet_id) = tagid.split(':')
    return tweet_id

def get_tweets_single(twapi, idfilepath):
    '''
    Fetches content for tweet IDs in a file one at a time,
    which means a ton of HTTPS requests, so NOT recommended.

    `twapi`: Initialized, authorized API object from Tweepy
    `idfilepath`: Path to file containing IDs
    '''
    # process IDs from the file
    with open(idfilepath, 'rb') as idfile:
        for line in idfile:
            tweet_id = get_tweet_id(line)
            Logger.debug('get_tweets_single: fetching tweet for ID %s', tweet_id)
            try:
                tweet = twapi.get_status(tweet_id)
                print('%s,%s' % (tweet_id, tweet.text.encode('UTF-8')))
            except tweepy.TweepError as te:
                Logger.warn('get_tweets_single: failed to get tweet ID %s: %s', tweet_id, te.message)
                # traceback.print_exc(file=sys.stderr)
        # for
    # with

def get_tweet_list(twapi, idlist):
    '''
    Invokes bulk lookup method.
    Raises an exception if rate limit is exceeded.
    '''
    # fetch as little metadata as possible
    tweets = twapi.statuses_lookup(id_=idlist, include_entities=False, trim_user=True)
    if len(idlist) != len(tweets):
        Logger.warn('get_tweet_list: unexpected response size %d, expected %d', len(tweets), len(idlist))
    for tweet in tweets:
        print('%s,%s' % (tweet.id, tweet.text.encode('UTF-8')))

def get_tweets_bulk(twapi, idfilepath):
    '''
    Fetches content for tweet IDs in a file using bulk request method,
    which vastly reduces number of HTTPS requests compared to above;
    however, it does not warn about IDs that yield no tweet.

    `twapi`: Initialized, authorized API object from Tweepy
    `idfilepath`: Path to file containing IDs
    '''
    # process IDs from the file
    tweet_ids = list()
    with open(idfilepath, 'rb') as idfile:
        for line in idfile:
            tweet_id = get_tweet_id(line)
            Logger.debug('Enqueing tweet ID %s', tweet_id)
            tweet_ids.append(tweet_id)
            # API limits batch size
            if len(tweet_ids) == batch_size:
                Logger.debug('get_tweets_bulk: fetching batch of size %d', batch_size)
                get_tweet_list(twapi, tweet_ids)
                tweet_ids = list()
    # process remainder
    if len(tweet_ids) > 0:
        Logger.debug('get_tweets_bulk: fetching last batch of size %d', len(tweet_ids))
        get_tweet_list(twapi, tweet_ids)

def err_msg():
    print('Usage: get_tweets_by_id.py [options] file')
    print('    -s (single) makes one HTTPS request per tweet ID')
    print('    -v (verbose) enables detailed logging')
    sys.exit()

def main(args):

    bulk = True
    try:
        opts, args = getopt.getopt(args, 'sv')
    except getopt.GetoptError:
        err_msg()
    for opt, _optarg in opts:
        if opt in ('-s'):
            bulk = False
        elif opt in ('-v'):
            pass
            # Logger.setLevel(logging.DEBUG)
            # Logger.debug("main: verbose mode on")
        else:
            err_msg()
    if len(args) != 1:
        err_msg()
    idfile = args[0]
    if not os.path.isfile(idfile):
        print('Not found or not a file: %s' % idfile, file=sys.stderr)
        err_msg()

    # connect to twitter
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)

    # hydrate tweet IDs
    if bulk:
        get_tweets_bulk(api, idfile)
    else:
        get_tweets_single(api, idfile)

if __name__ == '__main__':
    main(sys.argv[1:])