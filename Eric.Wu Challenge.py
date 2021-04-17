import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# read in raw dataset
eluvio = pd.read_csv("C:/Users/HOME/Desktop/excel_files/Eluvio_DS_Challenge.csv")


# reshape dataframe to a way we want
# dict1 contains author and the number of shows they've done, dict2 also contains author and the number of upvotes they've gotten
dict1 = {'author':eluvio['author'].value_counts().index.tolist(), '#oftitles':list(eluvio['author'].value_counts())}
dict2 = {'author':eluvio.groupby('author').sum('up_votes')['up_votes'].index.tolist(), 'upvotes':list(eluvio.groupby('author').sum('up_votes')['up_votes'])}
# converting them to dataframe, and then joining them together into one master dataframe
dict1_df = pd.DataFrame(data = dict1)
dict2_df = pd.DataFrame(data = dict2)
joined_df = dict1_df.merge(dict2_df, on='author', how='left')
# add an 'average' column, where we see the average number of upvotes they get per title
joined_df['average'] = round(joined_df['upvotes']/joined_df['#oftitles'],2)


joined_df = joined_df[joined_df['#oftitles'] >= 100] #want authors who have high sample size of shows (100 or more titles)
joined_df = joined_df[joined_df['average'] >= 300] #want authors who average upvotes of 300 or more per show
joined_df = joined_df.sort_values(by='average', ascending=False) #sort dataframe by highest to lowest average upvotes


# quick exploratory analysis to see how well authors with many shows and a high average upvote count compare with each other
authors_col = joined_df['author']
average_col = joined_df['average']
average_total = sum(average_col)
fig1, ax1 = plt.subplots(figsize=(10,10)) #want a large pie chart to graph the many authors
ax1.pie(average_col, labels=authors_col, autopct='%1.2f%%', pctdistance=0.9, startangle=90) #higher pctdistance for better visibility
ax1.axis('equal')
plt.legend(
    loc='best',
    labels=['%s, %i, %1.2f%%' % ( #name, average, percentage of pie
        l, a, (float(s) / average_total) * 100) for l, a, s in zip(authors_col, average_col, average_col)],
    bbox_to_anchor=(0.0, 1),
    bbox_transform=fig1.transFigure
)
plt.show()
# interestedin86 has the highest upvote to show ratio, trailed by toomanyairmiles, Vranak, and KRISHNA53


# Do authors with a large number of shows produce titles with a higher number of upvotes? Authors who have produced many shows should have a better understanding on what gets more attention/upvotes, and continue producing quality content. I want to see if my theory is true.
# Can answer question using linear regression model
# using X (number of titles) to predict y (average upvotes)
X = joined_df[['#oftitles']]
y = joined_df['upvotes']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

model.summary()

#model, y = B0 + B1*x, y = upvotes, B0 = constant, B1 = #oftitles, x = coefficient
#null hypothesis: the model is inadequate in predicting the number of upvotes
#alternative hypothesis: the model is good at predicting the number of upvotes
#test statistics: p-value = 2.24*10^-25
#rejection region: p-value < significance level alpha = 0.05
#conclusion: Because p-value is smaller than alpha, we can reject the null hypothesis, and conclude that our model is good at predicting the number of upvotes.

#conclusion: 
#the model is y = -1.561*10^4 + 465.7987*x
#the coefficient for the constant is -1.561*10^4
#the coefficient for #oftitles is 465.7987; for every unit increase of #oftitles, upvotes increases by 465.7987 

#because R-squared is a very high number (0.98), the model does very well in predicting our independent variable (upvotes)

