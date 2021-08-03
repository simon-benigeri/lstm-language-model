This directory contains data files in txt format used for the task. 

## Overview of Data

The New York Times Covid Dataset was collected by querying the New York Times API for the word "coronavirus". we searched through each page on a given day, starting with January 1, 2020. Some days had fewer than 10 related articles, especially towards the beginning, but later days often yielded dozens of results. Since each response contained a maximum of 10 results, any days with more than 10 results would need to be paged through repeatedly until all URLs had been discovered for that day

We scraped a total of 9302 unique URLs. The URLs scraped provided around 15M tokens, so we randomly selected enough articles for around 2.5M tokens.

Then we parse the html files using BeautifulSoup library. We extract articles from the collected html files and remove the image caption text in the process. And then we seperate each token with a space. The file contains every sentence ending with a </s> token.

The WikiText-2 data is a well known dataset. More information can be found here: https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
