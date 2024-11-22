# Notes
## GIT
### How can I figure out the difference between restore, revert, rebase, and reset?
#### Restore: seems to restore something that was deleted?
#### Revert: creates a new commit that undoes previous changes
#### Rebase: let's you edit history of commits
#### Reset: moves branch back in time to specified commit
### Git is case sensitive. I need to check for correctness separately for case sensitive answers
### Need a better definition for git retores? Maybe?

import random
import math

def rundeck():
    study_dict=tricky_dict

    rand_conf_keys=random.sample(list(confident_dict.keys()), math.ceil(len(confident_dict)*0.33))

    for key in rand_conf_keys:
        study_dict[key]= confident_dict[key]

    length_list=list(range(len(study_dict)))

    while len(length_list)>0:
        rand_num=random.choice(length_list)
        rand_key, rand_value = list(study_dict.items())[rand_num]
        print(rand_value)
        myinput=input()
        if myinput.upper().replace(' ','')==rand_key.upper().replace(' ',''):
            print('Correct!')
            length_list.remove(rand_num)
        elif myinput.upper()=="QUIT":
            quit()
        else:
            print('Incorrect. The correct answer is', rand_key)

subjlist=['PYTHON', 'PY', 'R', 'APPLIED STATISTICS IN R', 'ASIR',
          'PRACTICAL MACHINE LEARNING', 'PML', 'DATABASE SYSTEMS', 'DBS',
          'DECISION ANALYTICS', 'DA', 'GENERAL KNOWLEDGE', 'GK',
          'GIT']

studying='YES'

while studying in ['YES' , 'Y']:

    subject=''
    print('''Which subject would you like to study?:
          PYTHON (PY)
          R
          APPLIED STATISTICS IN R (ASIR)
          PRACTICAL MACHINE LEARNING (PML)
          DATABASE SYSTEMS (DBS)
          DECISION ANALYTICS (DA)
          GENERAL KNOWLEDGE (GK)
          GIT
    ''')
    while subject not in subjlist:
        subject=input().upper()
        if subject=='QUIT':
            quit()
        if subject not in subjlist:
            print('Sorry, that subject is not available. Please select another.')

    if subject in ['PYTHON', 'PY']:

        confident_dict={
        ".append()": "This method adds a single item to a list",
        ".extend()": "This method adds multiple items to a list",
        "del list[index]": "This deletes an item from a list based on index",
        "df.columns": "Creates a list of columns in a pandas data frame",
        "df['Column'].astype()": "Converts one data type to another for specified column(s). Specify a column in answer",
        "df.loc[]": "Used to access a group of rows and columns by label(s) or a boolean array. Let's you subset data frame using logicals.",
        "df.iloc[]": "Gets or sets value(s) at speicifed data frame indices",
        "df.describe()": "Provides summary statistics of numerical data columns",
        "df.sort_values()": "Sorts a data frame based on selected column(s)",
        "~": "Not operator in .loc method",
        ".add()": "Adds items to a set",
        # These take too long to type
        # "pd.DataFrame([('row1col1', 'row1col2'), ('row2col1', 'row2col2'), ('row3col1', 'row3col2')], columns=['col1','col2'])": "Create pandas dataframe using a list tuples. 3 rows 2 cols",
        # "pd.DataFrame({'col1': ['row1', 'row2', 'row3'], 'col2': ['row1', 'row2', 'row3']})": "Create pandas data frame using dictionary of lists",
        ".sort()": "Sorts in place and returns nothing",
        "sorted()": "Creates a returns a new sorted sequence",
        "df['Column'].hist": "Create histogram for a pandas series",
        "df.drop(columns=['Column'])": "Drops a column from the data frame",
        "df.sum(axis=)": "Sums selected rows/columns, depending on axis value",
        "df.rename(columns={'Oldname': 'Newname'})": "Renames column(s) in data frame. In answer, use 'Oldname' and 'Newname'",
        "df.reset_index(drop=True)": "Creates a new index for a subsetted data frame and eliminates old index",
        "df['Column'].str": "Allows you to perform string operations on each element of data frame. Specify a column in answer",
        "df.groupby(['Column'])": "Aggregates by a specified column. I think it would need to be followed by a function. Specify column",
        "'string'.join(iterable)": "Takes all items in an iterable and joins them into one string with string object as separator. Specify string and iterable",
        "pd.options.display.max_rows = 20": "Set display for a pandas data frame to a maximum of 20 rows",
        "names=[]": "Option in pandas read statement that will allow you to name the columns getting imported",
        "df.info()": "Provides basic information about the data frame",
        "df.dtypes": "Returns data types of columns in data frame",
        "df.column.values": "Produces a naked array of values in a column. Specify column using dot notation",
        "nobels[nobels.nobelist.str.contains('Curie')]": "Would return all rows from Nobels data frame where the nobelist's name includes 'Curie'. Dot notation.",
        'df.copy()': 'Creates a copy of a dataframe',
        'df.shape': 'Gives dimensions of your data frame',
        '.split()': 'Splits a string into a list',
        'matplotlib': 'Foundational library for visualizations. Many other viz libraries are built on top of this one',
        'seaborn': 'A statistical viz library built on top of matplotlib',
        '@': 'Numpy matrix multiplication',
        "df['Column'].value_counts()": "Creates a frequency for a column. Specify column.",
        'pd.concat([df1, df2])': 'Appends dataframes. Include df1 and df2 as the dataframes getting appended',
        'statsmodels': 'A library with many advanced statistical functions',
        'scipy': 'Library that provides advanced scientific computing, including functions for optimization, linear algebra, image processing and much more',
        'scikit-learn': 'Most popular machine learning library for Python (not deep learning)',
        "pd.merge(df1, df2, on=['Column'], how='left')": 'Left join two datasets. Specify dfs and assume primary key is same.',
        'dropna=False': 'Argument in value_counts() that will include NaN values',
        "df['Column'].fillna()": "Will fill NA values in a column with a specified value",
        'np.where(condition, x, y)': 'NumPy funciton that returns values from x or y depending on condition',
        "df.sort_index(ascending=False)": "Sort by index by descending",
        'df.corr()': 'Calculates correlations between columns',
        "df['Column'].nunique()": 'Number of unique rows in column',
        'normalize=True': 'Argument in value_counts() that gets percentage breakdown',
        "pd.to_datetime(df['Column']).dt.year": 'Get year part of pandas column that is not already in a recognizable datetime format',
        "f'Hello {name}. You are studying {language}.'": "An f-string that would generate 'Hello Mac. You are studying Python.' with Mac and Python assigned as variables",
        ".remove()": "This deletes an item from a list based on value",
        "df.set_index(['Column'])": "Make it so a certain column is now the index",
        '.replace()': 'Replaces a specified string with another specified string',
        '.index()': 'Returns the index of a value from a list',
        "df.drop_duplicates(subset=['Column'])": "Deduplicates a data frame by a column(s). Specify column.",
        'np.array([])': 'Create NumPy array',
        'array[[]]': 'Multi-index from array',
        'df.dropna()': 'Drops NA values',
        'np.zeros(size, dtype)': 'Empty numpy array. Enter hypothetical args',
        'np.zeros_like()': 'Empty numpy array in the shape of another array',
        'np.linspace(min, max, n)': 'Linearly spaced numpy array that includes number of data points specified',
        'np.arange(min, max, interval)': 'Numpy array with equally spaced specified intervals',
        'np.random.random(size=(8,8))': 'Random 8x8 numpy array',
        
        'lazy': 'lazy'
        }

        tricky_dict={
        "df.boxplot(column=['Column'])": "Creates a boxplot of a series",
        'np.argmax()': 'Returns the indices of the maximum values along an axis',

        'lazy': 'lazy'
        }

        rundeck()

    if subject=='R':

        confident_dict={
        "seq()": "Creates a sequence of numbers",
        "c()": "Creates a vector",
        "rm(list=ls())": "Clears environment",
        "str()": "Returns structure of object",
        "data.frame(column1= vector, column2= vector)": "Creates a data frame. Specify two columns",
        "expand.grid()": "This can be used to map all possible outcomes of a combination",
        "gtools::permutations()": "Calculates a permutation. Not from base R",
        "union()": "Finds the universe of values in the given vectors",
        "intersect()": "Finds values that occur in both vectors",
        "choose()": "Finds number of possible arrangements where order doesn't matter. No replacement I think",
        "apply(trees, 2, mean)": "Returns the average of each column in the trees data set",
        "d": "Probability function type: Returns a density estimate",
        "p": "Probability function type: Returns the probability to the left (less than or equal to) or to the right (greater than) of a given value or quantile",
        "q": "Probability function type: Returns the value or quantile for one (1) or more probabilities",
        "r": "Probability function type: Returns random values generated from the specified distribution",
        "colnames()": "Provides the names of columns in a data frame",
        "%*%": "Matrix multiplication",
        'chisq.test()': 'Performs a chi-square test',
        't.test()': 'Performs a t test',
        'addmargins()': 'Performs operations for each row and each column and adds results to margins of table',
        'cumsum()': 'Cumulative sum for frequency distribution',
        'cor.test()': 'Test for correlation',
        "dim()": "Gives dimensions",
        "append()": "Adds element(s) onto a vector",
        "Solve()": "Gets matrix inverse",



        'lazy': 'lazy'
        }

        tricky_dict={
        "name<- function(args){operations}": "Creates a function",
        "unique()": "Deduplicates a vector",
        "duplicated()": "Produces a Boolean vector of values that are duplicates of previously occurring values",
        "any()": "Is at least one of the values in a vector TRUE?",
        "aggregate()": "Aggregates by a specified columns(s)",
        "for (i in vector){statements}": "For loop",
        "if(test expression){statements}": "If statement",
        "nrow()": "Number of rows",
        "sample()": "Sample from a vector/data frame",
        "sample.int()": "Generates a sample from 1:n",
        "dplyr::sample_n()": "Directly samples rows from a data frame",
        "prop.test()": "Determines a confidence interval of a population proportion",
        'library()': 'Access an installed package',
        'moments and rockchalk': "Packages that you'd find skewness and kurtosis in",
        'ifelse()': 'If statment that allows for vectorized operations',
        'aov()': 'Performs analysis of variance',
        'TukeyHSD()': 'Performs test to determine which means of groups are significantly different',
        'cor()': 'Gets correlation of two variables',
        'corrplot()': 'Returns correlation matrix',
        'lm(y~x)': 'Performs regression',
        'with()': "So you don't have to keep referencing the name of a data frame",
        "vector[vector>0]": "Returns subset of vector where values are greater than 0",
        "sum(vector>0)": "Returns number of values in a vector that are greater than 0",

        'lazy': 'lazy'
        }

        rundeck()

    if subject in ['APPLIED STATISTICS IN R', 'ASIR']:
        confident_dict={
        'Inferential': "If data are gathered from a subgroup of a larger group and the data are used to reach conclusions about the larger group, then the statistics are said to be ________ statistics.",
        'Parameter': 'Constant measure of population',
        'Statistic': 'Variable measure of sample',
        'Nominal': 'Categorized level of data with no order',
        'Ordinal': 'Categorized level of data with order',
        'Interval': 'Numerical level of data where distances between consecutive numbers have meaning, but there is no true 0',
        'Ratio': 'Numerical level of data where distances between consective numbers have meaning and the 0 represents the absence of the characteristic being studied',
        'Relative frequency': 'Proportion of total frequency at any given interval',
        'Binomial': 'A discrete distribution where each trial only has two possible outcomes denoted as success or failure',
        'Poisson': 'A discrete distribution that describes discrete occurrences over a continuum or interval',
        'Hypergeometric': 'A discrete distribution measures successes vs. failures but does not use replacement',
        'Uniform': 'A continuous distribution in which the same height is observed over a range of values',
        'Normal': 'A continuous distribution that is symmetrical about its mean and unimodal',
        'Exponential': 'A continuous distribution that describes the times between random occurrences',
        'Sample means': 'According to the Central Limit Theorem, what values will be approximately normally distributed at a sufficiently large sample size?',
        '30': 'Accepted as the sample size for Central Limit Theorem to apply',
        'Population mean': 'According to the Central Limit Theorem, the mean of the sample means will be equal to what at a sufficiently large sample size?',
        'Popstdev/sqrt(n)': 'According to the Central Limit Theorem, the standard deviation of the sample means will be equal to what at a sufficiently large sample size?',



        'lazy': 'lazy'
        }

        tricky_dict={
        'Inductive': "Another name for inferential statistics",
        'Chi-Squared Goodness of Fit Test': 'Test used to analyze probabilities of multinomial distributions along a single dimension. Say test',
        'Chi-Squared Test of Independence': 'Test used to analyze frequencies of two varaibles with multiple categories to determine if they are independent',
        '5': 'In using the chi-square goodness-of-fit test, a statistician needs to make certain that none of the expected values are less than ____',
        'columntotal*rowtotal/totaltotal': 'Calculate the expected values of a contingency table (aka a two-way frequency table). Use alphabetical order',
        'Interval or ratio': 'Data level required for parametric statistics',
        'Ogive': 'Chart that displays cumulative frequncies of a dataset',
        'Pareto': 'Chart that shows frequency in bar form along with a cumulative percentage line',
        "Chebyshev's Theorem": "Theorem find the proportion of values within k standard deviations of the mean regardless of distribution. At least 1-(1/k^2) values will fall within +/-k standard deviations of mean.",
        "100*stdev/mean": "Formula for coefficient of deviation",
        'Sample space': 'Complete list of all elementary events for an experiment',
        'Elementary event': 'An event that cannot be decomposed or broken down into other events',
        'Mutually exclusive': 'When two sets have no common outcomes. Occurrence of one event precludes the occurrence of another',
        'Independent': 'If the occurrence of one event does not affect the occurrence or nonoccurrence of the other event',
        'Collectively exhaustive': 'If a set contains all possible elementary events for an experiment, it is said to be _____',
        'Marginal': 'Word describing the probability of achieving just one event',
        'P(X U Y) = P(X) + P(Y) - P(X n Y)': 'General law of addition for probability',
        "Mutually exclusive 2": 'P(X U Y)= P(X) + P(Y) when the events are _______. Add a 2',
        'P(X n Y) = P(X) * P(Y|X)': "General law of multiplication for probability. Use P(X) in answer",
        'Independent 2': 'P(X n Y) = P(X) * P(Y) when the events are _______. Add a 2',
        'Lambda': 'Long-run average in poisson distribution',
        'Change lambda': 'If lambda and x intervals are different in a given problem, you should _______',
        'n>20 and n*p<7': 'If these criteria are met, poisson distribution can be used to approximate binomial distribution',
        'n*p>5 and n*q>5': 'If these criteria are met, normal distribution can be used to approximate binomial',
        '0.5': 'To correct for continuity when using a discrete distribution to approximate a continuous, this number must be added or subtracted',
        'Also n*p>5 and n*q>5': "Criteria for Central Limit Theorem to apply to sample proportions. Prefix with 'Also'",
        'One sample t test': 'This test is used when there is one sample and it is less than 30 and population standard deviation is unknown',
        'Chi-squared': 'Distribution used to estimate population variance based on sample variance',
        'Type 1': 'Which error rejects a true null hypothesis',
        'Type 2': 'Which error fails to reject a false null hypothesis',
        'Alpha': 'Probability of committing a Type 1 error is equal to _____',
        'Confidence': 'Equal to 1 - alpha',
        'Beta': 'Probability of committing a Type 2 error is equal to _____',
        'Power': 'Equal to 1 - beta',
        'Finite correction factor': 'It seems like this has to be used for a z test where the sample is equal to or greater than 5% of pop',
        'Equal population variances': 'For t-tests involving two samples, in order to determine which t-test to use, you need to determine if you have ________',
        'Dependent': 'Another factor that determines which t-test to use is if the samples are ______. Also referred to as pooled sample t-test',
        'Z': 'Statistic for testing proportions',
        'F': 'Type of distribution two sample variances follow',
        'One way ANOVA': 'Test used when there are 3 or more samples and 1 independent variable',
        'F 2': 'Test used to determine signficance of ANOVA test. Add 2 to the end',
        'Tukey tests': 'Multiple comparison tests used to determine which mean is different from others. Two different tests depending on whether the sample sizes are the same. Say tests',
        'Blocking': 'A variable used in ANOVA tests to control for confounding or concomitant variables',
        'Two Way ANOVA': 'Test used when there are 3 or more samples and 2 or more independent variables',
        'Interaction': 'This occurs in two way ANOVA when the effects of one treatment vary according to the levels of another treatment',
        'Degrees of freedom': 'In order to find the values in the chi-square distribution table, you must convert the sample size to _________',
        'F 3': 'Test used to determine the signficance of a multiple regression model. Add 3',




        'lazy': 'lazy'
        }

        rundeck()


    if subject in ['PRACTICAL MACHINE LEARNING', 'PML']:

        confident_dict={
        'Supervised': 'Type of learning that involves building a statistical model for predicting an output based on one or more inputs.',
        'Unsupervised': 'Type of learning that has inputs but no explicit outputs. Used for determining relationships and structure',
        'Semi-Supervised': 'Type of learning that has some labeled instances and some unlabeled instances',
        'Reinforcement': 'Type of learning that involves no training data. Agent learns from rewards/penalties',
        "Cross validation": 'Process involving splitting training data into training and validation set. Then using these to create and validate model',




        'lazy': 'lazy'
        }

        tricky_dict={
        'Batch': 'Type of training where system is trained then launched into production without continuing to learn',
        'Online': 'Type of training where system continues to learn as it runs',
        'Generalization': 'Word that describes a models ability to predict unseen data',
        'Instance-based': 'Learning method where agent learns examples and then generalizes to new cases by using a similarity measure to compare',
        'Model-based': 'Learning method where agent uses set of examples to build model and uses said model to make predictions',
        'Inference': "Type of analysis where goal is to understand relationship and predictors. F(x) can't be treated as a black box",
        'Parametric': 'Learning method that reduces problem of estimating f(x) down to one of estimating a set of parameters',
        'Nonparametric': 'Learning method that does not make assumptions about the functional form of f(x). Instead, seek estimate of f(x) that gets close to data points as possible without getting too rough or wiggly',
        'Overfitting': 'Following training data too closely and lacking generalizability. Model is overly flexible',
        'Flexible': 'If a model can fit many different possible functional forms for f(x), it is said to be ________',
        'Variance of fhat': 'The amount by which fhat would change if we estimated it using a different training set',
        'Squared bias of fhat': 'Error in a model introduced by approximating a real life problem using a simple model',
        'Higher': 'More flexibility results in _____ variance',
        'Lower': 'More flexibilty results in ______ bias',
        'Bayes classifier': 'This classifies a response based on the highest',
        'K-nearest neighbors': 'Classification method that involves classifying a response based on the other responses it is closest to',
        'Validation': "A set known as a 'hold out'. Used to test accuracy of model before applying to test, which we won't know correct answers for",
        'Bootstrap': 'Statistical tool that repeatedly samples observations from the original dataset to quantify the uncertainty associated with a given estimator or statistical learning method',
        'Step': 'Type of function that cuts the range of a variable into K distinct regions in order to produce a qualitative variable',
        'Regression splines': 'Divides range of X into K distinct regions and fits a polynomial function within each region',
        'Knots': 'Divided regions in a spline that are fit to a polynomial function',
        'Smoothing splines': 'Divides range of X into K distinct regions and fits a polynomial function within each region and subjects to additional smoothing penalty',
        'Local Regression': 'Similar to splines, but regions are allowed to overlap',
        'Generalized Additive Model': 'I believe it is a single regression model made up of several regression models',
        'Batch gradient descent': 'Full name. Type of gradient descent where you sweep through entire dataset',
        'Stochastic gradient descent': 'Full name. Type of gradient descent where parameters are selected at random and error is minimized for a random subset of data iteratively',
        'Epoch': 'When entire dataset has passed forward and backward through the neural network once',
        'Ridge Regression': 'Full name. Regression regularization technique where weights are constrained so that they are close to 0. Uses L2 regularization.',
        'Lasso Regression': 'Full name. Aggressive regression regularization technique that uses L1 regularization and some weights can actually be 0.',
        'Elastic Net': 'Regularization technique that is somewhere between Lasso and Ridge.',
        'Support Vector Machine': 'Classifier method that creates decision boundary by maximizing distance between classes on each side of boundaries',
        'Kernal Trick': 'A method of feature tranforming that allows for data to be linearly seperable. Used in SVM',
        'Recall/Sensitivity/TPR': 'Alphabetic. True positives divided by true positive and false negative. True positives divided by actual positives. Ex: Test detects 99% of COVID',
        'Specificity/TNR': 'Alphabetic. True negatives divided by true negatives and false positives. True negatives divided by actual negatives.',
        'Precision': 'Alphabetic. True positive divided by true positives and false positives. True positive divided by predicted positive. Ex. When test says COVID, it is right 99% of the time',
        'ROC Curve': 'Graph that measures true positives (y-axis) vs. false positives (x-axis). Should be pulled up to the left',
        'Precision-Recall Tradeoff': 'Graph that displays the inverse relationship between precision and recall',
        'One-versus-the-rest': 'Method of multiclass classification where a bunch of binary classifiers are trained. Ex. Image is either a 5 or it is not',
        'One-versus-one': 'Method of multiclass classification where each pair of groups has a classifier. Ex. 0v1, 0v2, 1v2, etc.',
        'C': 'Hyperparameter (at least in Python) that sets the hardness of classification in SVM',
        'Kernal trick': 'Method that increases dimensionality of a model to find linear separatation',
        'Purity': 'Measure of homogeniety in a decision tree node',
        'Entropy and gini': 'Specific metrics of purity in decision tree. Alphabetic',
        'Random Forest': 'Type of classifier that uses a group of decision trees modeled off of different subsets of the data',
        'Bagging': 'An ensemble decision tree method that involves using a subset of the data with replacement',
        'Pasting': 'An ensemble decision tree method that involves using a subset of the data without replacement',
        'Boosting': 'A strategy with trees where several weak learners are combined into a strong learner',
        'Stacking': 'Strategy with multiple models where, instead of using hard voting, a model is trained to aggregate',
        'Principal Component Analysis': 'Method of dimension reduction. Identifies hyperplane that lies closes to the data, then projects data onto it. Selects hyperplane that preserves maximum variance and is most likely to lose less info than others.',
        'Projection': 'Method of dimension reduction that simply flattens data down into a lower dimension. Like 3D to 2D',
        'Manifold Learning': 'Method of dimension reduction that unbends and untwists down to a lower dimension',
        'K-Means': 'Method of clustering that identifies centroids and classifies instances based on this',
        'DBSCAN': 'Method of clustering that is better for clustering weird shapes',
        'ReLU': 'Most common activation function for a neuron',
        'Batch Normalization': 'Used in neural network to address exploding/vanishing gradients. Performed at every hidden layer',
        'Stride': 'How much filter is moved in CNN',
        'Pooling': 'Feature reduction in CNN. For example, taking max value in a region and only keeping that',
        'Dropout Layer': 'A layer in CNN that drops a certain percentage of neurons to simplify',




        'lazy': 'lazy'
        }

        rundeck()


    if subject in ['DBS', 'DATABASE SYSTEMS']:

        confident_dict={
        'DBMS': 'A generalized software system for manipulating databases',


        'lazy': 'lazy'
        }

        tricky_dict={
        'Data Independence': 'The ability to make changes in either the logical or physical structure of the database without requiring reprogramming of application programs',
        'Functional Dependency': 'Term for when knowing one value means you know another',
        'Deletion anomoly': 'Occurs when deleting info you no longer want also removes info you want to keep',
        'Insertion anomoly': 'Occurs when you want to add information to a table, but table has multiple set of facts, so you need to add additional, unnecessary facts',
        'Normalization': 'Process of breaking up table with multiple sets of facts into multiple tables',
        'Domain/Key Normal Form': 'Only normal form proven to be anomoly free. Defined by if every constraint on relation is a logical consequence of the definitions of keys and domains',
        'Constraint': 'Rule that governs the static value of attributes',
        'Domain': 'Set of values that an attribute can assume',
        'Not null': 'Type of constraint that assures a null value cannot be stored in a column of a database table',
        'Unique': 'Type of constraint that specifies that duplicate values are not allowed in a column',
        'Primary key': 'Type of constraint that specifies that every row in a table is distinct from all other rows',
        'Foreign key': 'Type of constraint that identifies a column as being linked to the primary key of another table in the database',
        'Check': 'Type of constraint that validates the value entered into the column against a specified condition (ex. non-negative)',
        'Default': 'Type of constraint that specified a default value for a column if no value is specified',


        'lazy': 'lazy'
        }

        rundeck()

    if subject in ['DA', 'DECISION ANALYSIS']:

        confident_dict={

        'lazy': 'lazy'
        }

        tricky_dict={
        'Reduced Cost': 'In sensitivity analysis, this shows how much the objective function coefficients can be increased or decreased before the optimal solution changes',
        'Shadow Price': 'In sensitivity analysis, this shows how much the opimtal solution can be increased or decreased if we change the right hand side values (resources available) with one unit',
        'Allowable Increase': 'In sensitivity analysis, this is the amount by which the objective function can increase without changing the optimal solution',
        'Upper Bound': 'Assuming the primal is a maximization problem then a feasible solution to the dual provides a(n) _____ _____ for the objective value of the primal',
        'Lower Bound': 'Assuming the primal is a maximization problem then a feasible solution to the primal provides a(n) _____ _____ for the objective value of the dual',
        'Infeasible': 'If the objective function value to the primal is unbounded, then the dual is _____',
        'Minus': 'The reduced cost for a decision variable is the per unit profits _____ the per unit cost',
        'Right hand side': 'The solution to an LP problem is degenerate if the ________ of any of the constraints have an allowable increase or an allowable decrease of 0',



        'lazy': 'lazy'
        }

        rundeck()


    if subject in ['GENERAL KNOWLEDGE', 'GK']:

        notchecking_dict={
        'Popstdev/sqrt(n)': 'According to the Central Limit Theorem, the standard deviation of the sample means will be equal to what at a sufficiently large sample size?',
        'Relative frequency': 'Proportion of total frequency at any given interval',
        'Inductive': "Another name for inferential statistics",
        '5': 'In using the chi-square goodness-of-fit test, a statistician needs to make certain that none of the expected values are less than ____',
        'columntotal*rowtotal/totaltotal': 'Calculate the expected values of a contingency table (aka a two-way frequency table). Use alphabetical order',
        'Interval or ratio': 'Data level required for parametric statistics. Alphabetical',
        'Ogive': 'Chart that displays cumulative frequncies of a dataset',
        'Pareto': 'Chart that shows frequency in bar form along with a cumulative percentage line',
        'n>20 and n*p<7': 'If these criteria are met, poisson distribution can be used to approximate binomial distribution',
        'n*p>5 and n*q>5': 'If these criteria are met, normal distribution can be used to approximate binomial',
        '0.5': 'To correct for continuity when using a discrete distribution to approximate a continuous, this number must be added or subtracted',
        "100*stdev/mean": "Formula for coefficient of deviation",
        'Also n*p>5 and n*q>5': "Criteria for Central Limit Theorem to apply to sample proportions. Prefix with 'Also'",
        'F': 'Type of distribution two sample variances follow',
        'F 2': 'Test used to determine signficance of ANOVA test. Add 2 to the end',
        'F 3': 'Test used to determine the signficance of a multiple regression model. Add 3',
        'Z': 'Statistic for testing proportions',
        'Variance of fhat': 'The amount by which fhat would change if we estimated it using a different training set',
        'Squared bias of fhat': 'Error in a model introduced by approximating a real life problem using a simple model',
        'Ridge Regression': 'Full name. Regression regularization technique where weights are constrained so that they are close to 0. Uses L2 regularization.',
        'Lasso Regression': 'Full name. Aggressive regression regularization technique that uses L1 regularization and some weights can actually be 0.',
        'Elastic Net': 'Regularization technique that is somewhere between Lasso and Ridge.',
        'One-versus-the-rest': 'Method of multiclass classification where a bunch of binary classifiers are trained. Ex. Image is either a 5 or it is not',
        'One-versus-one': 'Method of multiclass classification where each pair of groups has a classifier. Ex. 0v1, 0v2, 1v2, etc.',
        'Not null': 'Type of constraint that assures a null value cannot be stored in a column of a database table',
        'Unique': 'Type of constraint that specifies that duplicate values are not allowed in a column',
        'Primary key': 'Type of constraint that specifies that every row in a table is distinct from all other rows',
        'Foreign key': 'Type of constraint that identifies a column as being linked to the primary key of another table in the database',
        'Check': 'Type of constraint that validates the value entered into the column against a specified condition (ex. non-negative)',
        'Default': 'Type of constraint that specified a default value for a column if no value is specified',


        'lazy': 'lazy'
        }

        confident_dict={
        'Inferential': "If data are gathered from a subgroup of a larger group and the data are used to reach conclusions about the larger group, then the statistics are said to be ________ statistics.",
        'Parameter': 'Constant measure of population',
        'Statistic': 'Variable measure of sample',
        'Nominal': 'Categorized level of data with no order',
        'Ordinal': 'Categorized level of data with order',
        'Interval': 'Numerical level of data where distances between consecutive numbers have meaning, but there is no true 0',
        'Ratio': 'Numerical level of data where distances between consective numbers have meaning and the 0 represents the absence of the characteristic being studied',
        'Binomial': 'A discrete distribution where each trial only has two possible outcomes denoted as success or failure',
        'Poisson': 'A discrete distribution that describes discrete occurrences over a continuum or interval',
        'Hypergeometric': 'A discrete distribution measures successes vs. failures but does not use replacement',
        'Uniform': 'A continuous distribution in which the same height is observed over a range of values',
        'Normal': 'A continuous distribution that is symmetrical about its mean and unimodal',
        'Exponential': 'A continuous distribution that describes the times between random occurrences',

        "Chebyshev's Theorem": "Theorem find the proportion of values within k standard deviations of the mean regardless of distribution. At least 1-(1/k^2) values will fall within +/-k standard deviations of mean.",
        'Sample space': 'Complete list of all elementary events for an experiment',
        'Elementary event': 'An event that cannot be decomposed or broken down into other events',
        'Change lambda': 'If lambda and x intervals are different in a given problem, you should _______',
        'Confidence': 'Equal to 1 - alpha',
        'Power': 'Equal to 1 - beta',
        'Finite correction factor': "It seems like this has to be used for a z test where the sample is equal to or greater than 5% of pop",
        'Equal population variances': 'For t-tests involving two samples, in order to determine which t-test to use, you need to determine if you have ________',
        'Dependent': 'Another factor that determines which t-test to use is if the samples are ______. Also referred to as pooled sample t-test',
        'Blocking': 'A variable used in ANOVA tests to control for confounding or concomitant variables',
        'Interaction': 'This occurs in two way ANOVA when the effects of one treatment vary according to the levels of another treatment',
        'Degrees of freedom': 'In order to find the values in the chi-square distribution table, you must convert the sample size to _________',

        'Batch': 'Type of training where system is trained then launched into production without continuing to learn',
        'Online': 'Type of training where system continues to learn as it runs',
        'Generalization': 'Word that describes a models ability to predict unseen data',
        'Instance-based': 'Learning method where agent learns examples and then generalizes to new cases by using a similarity measure to compare',
        'Model-based': 'Learning method where agent uses set of examples to build model and uses said model to make predictions',
        'Inference': "Type of analysis where goal is to understand relationship and predictors. F(x) can't be treated as a black box",
        'Parametric': 'Learning method that reduces problem of estimating f(x) down to one of estimating a set of parameters',
        'Nonparametric': 'Learning method that does not make assumptions about the functional form of f(x). Instead, seek estimate of f(x) that gets close to data points as possible without getting too rough or wiggly',
        'Overfitting': 'Following training data too closely and lacking generalizability. Model is overly flexible',
        'Validation': "A set known as a 'hold out'. Used to test accuracy of model before applying to test, which we won't know correct answers for",
        'Bootstrap': 'Statistical tool that repeatedly resamples observations with replacement from the original dataset to quantify the uncertainty associated with a given estimator or statistical learning method',
        'Step': 'Type of function that cuts the range of a variable into K distinct regions in order to produce a qualitative variable',
        'Regression splines': 'Divides range of X into K distinct regions and fits a polynomial function within each region',
        'Knots': 'Divided regions in a spline that are fit to a polynomial function',
        'Smoothing splines': 'Divides range of X into K distinct regions and fits a polynomial function within each region and subjects to additional smoothing penalty',
        'Local Regression': 'Similar to splines, but regions are allowed to overlap',
        'Generalized Additive Model': 'I believe it is a single regression model made up of several regression models',
        'Batch gradient descent': 'Full name. Type of gradient descent where you sweep through entire dataset using subcollections of the data',
        'Stochastic gradient descent': 'Full name. a type of gradient descent where model parameters are updated iteratively based on the gradient of the error calculated for a single randomly selected data point',
        'Epoch': 'When entire dataset has passed forward and backward through the neural network once',
        'C': 'Hyperparameter (at least in Python) that sets the hardness of classification in SVM',
        'Kernal trick': 'Method that increases dimensionality of a model to find linear separatation',
        'Purity': 'Measure of homogeniety in a decision tree node',
        'Entropy and gini': 'Specific metrics of purity in decision tree. Alphabetic',
        'Random Forest': 'Type of classifier that uses a group of decision trees modeled off of different subsets of the data',
        'Bagging': 'An ensemble decision tree method that involves using a subset of the data with replacement',
        'Pasting': 'An ensemble decision tree method that involves using a subset of the data without replacement',
        'Boosting': 'A strategy with trees where several weak learners are combined into a strong learner',
        'Stacking': 'Strategy with multiple models where, instead of using hard voting, a model is trained to aggregate',
        'Projection': 'Method of dimension reduction that simply flattens data down into a lower dimension. Like 3D to 2D',
        'ReLU': 'Most common activation function for a neuron',
        'Batch Normalization': 'Used in neural network to address exploding/vanishing gradients. Performed at every hidden layer',
        'Stride': 'How much filter is moved in CNN',
        'Pooling': 'Feature reduction in CNN. For example, taking max value in a region and only keeping that',
        'Dropout Layer': 'A layer in CNN that drops a certain percentage of neurons to simplify',

        'Data Independence': 'The ability to make changes in either the logical or physical structure of the database without requiring reprogramming of application programs',
        'Functional Dependency': 'DBS term for when knowing one value means you know another',
        'Deletion anomoly': 'Occurs when deleting info you no longer want also removes info you want to keep',
        'Insertion anomoly': 'Occurs when you want to add information to a table, but table has multiple set of facts, so you need to add additional, unnecessary facts',
        'Normalization': 'Process of breaking up table with multiple sets of facts into multiple tables',
        'Domain/Key Normal Form': 'Only normal form proven to be anomoly free. Defined by if every constraint on relation is a logical consequence of the definitions of keys and domains',
        'Constraint': 'DBS rule that governs the static value of attributes',
        'Domain': 'Set of values that an attribute can assume in database',

        '30': 'Accepted as the sample size for Central Limit Theorem to apply',
        'Marginal': 'Word describing the probability of achieving just one event',
        'Mutually exclusive': 'When two sets have no common outcomes. Occurrence of one event precludes the occurrence of another',
        'Independent': 'If the occurrence of one event does not affect the occurrence or nonoccurrence of the other event',
        'Lambda': 'Long-run average in poisson distribution',
        'Alpha': 'Probability of committing a Type 1 error is equal to _____',
        'Beta': 'Probability of committing a Type 2 error is equal to _____',
        'Flexible': 'If a model can fit many different possible functional forms for f(x), it is said to be ________',
        'Bias-Variance Tradeoff': "Balance between a model's ability to generalize well to new data and its ability to fit the training data closely, where increasing one often leads to increasing the other",
        'SVM': 'Classifier method that creates decision boundary by maximizing distance between classes on each side of boundaries. Use abbreviation',
        'Bayes classifier': 'This classifies a response based on the highest probability',
        'K-nearest neighbors': 'Classification method that involves classifying a response based on the other responses it is closest to',
        'ROC Curve': 'Graph that measures true positives (y-axis) vs. false positives (x-axis). Should be pulled up to the left',
        'K-Means': 'Method of clustering that identifies centroids and classifies instances based on this',
        'On-policy': 'Type of RL algo that learns the value of the policy it currently follows, often exploring and improving the same policy. Learns on the job',
        'Off-policy': 'Type of RL algo that learns the value of one policy while following a different behavior policy for exploration. Evaluates target policy while following behavior policy',
        'Perfect': 'Dynamic programming is a collection of algos that can be used to compute optimal policies given a ______ model of the environment',
        'Spatial correlations': 'In a CNN, a filter is intended to find ______ ______ in the input matrix',
        'Sequential': 'RNN is meant to capture _______ correlation',
        'Cross-validation': 'Resampling and sample splitting methods that use different portions of the data to test and train a model on different iterations',
        'PCA 2': 'Should perfrom this before running a decision tree. Add 2',
        'Sample means': 'According to the Central Limit Theorem, what values will be approximately normally distributed at a sufficiently large sample size?',
        'Population mean': 'According to the Central Limit Theorem, the mean of the sample means will be equal to what at a sufficiently large sample size?',
        'Collectively exhaustive': 'If a set contains all possible elementary events for an experiment, it is said to be _____',
        'P(X U Y) = P(X) + P(Y) - P(X n Y)': 'General law of addition for probability. What does the union of P(X) and P(Y) equal?',
        "Mutually exclusive 2": 'P(X U Y)= P(X) + P(Y) when the events are _______. Add a 2',
        'P(X n Y) = P(X) * P(Y|X)': "General law of multiplication for probability. What does the intersection of P(X) and P(Y) equal? Use P(X) in answer",
        'Independent 2': 'P(X n Y) = P(X) * P(Y) when the events are _______. Add a 2',
        'Chi-squared distribution': 'Distribution used to estimate population variance based on sample variance. Also used to test the independence of categorical variables and the goodness of fit between observed and expected distributions. Add "distribution" on end',
        'Higher': 'More flexibility results in _____ variance',
        'Lower': 'More flexibilty results in ______ bias',
        'Bias': 'In ML, refers to the error caused by a model making overly simplistic assumptions about the data',
        'Variance': 'In ML, refers to the error caused by a model being too sensitive to fluctuations in the training data',
        'Precision-Recall Tradeoff': 'Graph that displays the inverse relationship between precision and recall',
        'PCA': 'Method of dimension reduction. Identifies hyperplane that lies closes to the data, then projects data onto it. Selects hyperplane that preserves maximum variance and is most likely to lose less info than others. Use abbreviation',
        'LDA': 'Projects data onto a lower-dimensional space, maximizing the separation between different classes (primarily used for classification). Use abbreviation',
        'Manifold Learning': 'Method of dimension reduction that unbends and untwists down to a lower dimension',
        'Silhouette score': 'Calculation to determine number of clusters to use for k-means',
        'DBSCAN': 'Method of clustering that is better for clustering weird shapes',
        'Hierarchical Clustering': 'Method of clustering that computes distances of all possible pairs. Takes a very long time! Include "clustering"',
        'Decision Tree': 'Breaks down data into smaller subsets while at each node selects the feature that best splits the data based on criteria',
        'Gradient Boosting': 'Sequentially builds trees to correct the errors of previous ones',
        'Bootstrapping': 'In RL, the ability to learn before completing episodes',
        'Policy': 'In RL, maps states to actions',
        'Value-Based': 'Type of RL algo that focuses on learning the value function to determine the best actions',
        'Policy-Based': 'Type of RL algo that learns a policy directly, mapping states to actions without using a value function',
        'Dynamic Programming': 'Policy evaluation, policy iteration, and value iteration are all model-based methods, but more specifically, they are examples of ______',
        'Equal': "In a Markov Chain, the probability of next state given all previous states is _____ to probability of next state given current state",
        'Q-Learning': 'Model-free, value-based RL algo that is used to find the optimal action-selection policy using a q-function for any MDP. Off policy',
        'Deep Q-Learning': 'Replaces Q-table with a neural network',
        'Monte Carlo Learning': 'Learn directly from experience collected by interacting with environment. Sample episodes of experience and make updates to estimates at the end of each episode. Say "learning"',
        'Actor-critic method': 'A multi-agent RL method where one agent takes action based on policy distribution and one uses value function to inform other agent how good action was',
        'Free, Value, Off': "A Q-Learning algorithm is model-_____, ______-based, and ___-policy. Don't use 'and' in answer",


        'lazy': 'lazy'
        }

        tricky_dict={
        'Chi-Squared Goodness of Fit Test': 'Test used to determine whether the observed frequency distribution of a categorical dataset differs significantly from an expected distribution',
        'Chi-Squared Test of Independence': 'Test used to determine if there is a significant association between two categorical variables in a contingency table',
        'One-sample t-test': 'This test is used to determine whether the mean of a single sample is significantly different from a known or hypothesized population mean, especially when the population standard deviation is unknown',
        'Paired Sample t-Test': "Used to compare the means of two related groups to determine if there is a significant difference between them (e.g., before-and-after measurements)",
        'Independent Samples t-Test': "Used to compare the means of two independent groups to determine if they are significantly different",
        'One-way ANOVA': 'Test used to determine if the means of three or more groups differ significantly based on one independent variable',
        'Two-Way ANOVA': 'Test used to determine the effect of two independent variables and their interaction on a dependent variable',
        'Tukey tests': 'Post-hoc tests used after ANOVA to identify which specific group means differ, with variations depending on whether sample sizes are equal. Say tests',

        'Example of Chi-Squared Goodness of Fit Test': 'Example of... Suppose a candy company claims that their bags contain an equal proportion of five different colors of candies. You open a bag, count the candies, and compare the frequencies',
        'Example of Chi-Squared Test of Independence': 'Example of... A researcher wants to determine if candy color and preference (like/dislike) are independent of each other. They survey 100 people, and the responses are compared in a contingency table',
        'Example of One-sample t-test': 'Example of... The candy company claims the average number of candies per bag is 50. You take a sample of bags and want to test if the average in your sample differs from this claim',
        'Example of Paired Sample t-Test': 'Example of... You test a group of people for their preference rating of a candy before and after changing its packaging design (on a scale of 1-10)',
        'Example of Independent Samples t-Test': 'Example of... You want to compare the average preference rating (1-10) of two groups of people, one who tried a sugar-free candy and the other who tried the regular version',
        'Example of One-Way ANOVA': 'Example of... You want to compare the average sweetness levels (on a scale of 1 to 10) of three candy brands: Brand A, Brand B, and Brand C',
        'Example of Two-Way ANOVA': 'Example of... You want to study the effect of candy color (red, blue, green) and sugar type (natural, artificial) on sweetness levels',
        'Example of Tukey Tests': 'Example of... After performing an ANOVA, you find that there is a significant difference in sweetness levels among three candy brands. Now how do you determine which specific brands differ?',

        'Type 1': 'Which error rejects a true null hypothesis',
        'Example of Type 1': 'Example of... If a pregnancy test says a woman is pregnant when she is not, it would be considered this type of error',
        'Type 2': 'Which error fails to reject a false null hypothesis',
        'Example of Type 2': 'Example of... If a pregnancy test says a woman is not pregnant when she is, it would be considered this type of error',

        'Recall/Sensitivity/TPR': 'Alphabetic. True positives divided by true positive and false negative. True positives divided by actual positives. Ex: Test detects 99% of COVID',
        'Specificity/TNR': 'Alphabetic. True negatives divided by true negatives and false positives. True negatives divided by actual negatives.',
        'Precision': 'Alphabetic. True positive divided by true positives and false positives. True positive divided by predicted positive. Ex. When test says COVID, it is right 99% of the time',

        'Monte Carlo Simulation': 'Run multiple simulations based on random sampling to model complex environments',

        'lazy': 'lazy'
        }

        rundeck()


    if subject=='GIT':

        confident_dict={


        'lazy': 'lazy'
        }

        tricky_dict={
        'git config --global user.name "John Doe"': 'How would you assign John Doe as the active user',
        'git config --global user.email "johndoe@gmail.com"': 'How would you assign johndoe@gmail.com as the active email',
        'git init': 'Initializes a repo for the active directory',
        'git add <filename>': 'Add a specific file to the staging environment',
        'git add .': 'Add all files in active directory to staging environment',
        'git commit -m "Message"': 'Commit staged files to branch. Include a message that reads "Message"',
        'git log --oneline': 'View log of just commits',
        'git restore <filename>': 'Restore a singular file',
        'git restore .': 'Restore entire active directory',
        'git rm <filename>': 'Delete file. This change gets automatically moved into staging',
        'git mv <oldname> <newname>': 'Rename file. This change gets automatically moved into staging',
        'git diff': 'Show difference between two files',
        'git commit --amend -m "New Message"': 'Modify existing commit without creating a new one. Include a message that reads "New Message"',
        'git reset <hash>': 'Moves the branch to the specified commit. By default (--mixed), it unstages changes made after the specified commit but keeps them in the working directory',
        'git reset --hard <hash>': 'Moves the branch to the specified commit and resets the staging area and working directory to match that commit. Any changes made after the specified commit are discarded',
        'git rebase -i <branch>/<commit>': 'Lets you interactively adjust the history of your branch by applying its commits onto a specified <branch> or <commit> with options to modify or organize them',
        'git rebase -i HEAD~<#>': 'Allows interactive editing of the last # commits in your current branch, enabling actions like squashing, editing, reordering, or dropping commits',
        'git branch': 'Look at branches',
        'git switch -c <newbranch>': 'Copy current branch and create new branch named "newbranch"',
        'git merge <branch>': 'Merge another branch into current branch',
        'git branch -d <branch>': 'Deletes branch as long as there are no conflicts',
        'git branch -D <branch>': 'Forces git to ignore conflicts and just deletes branch',
        'git stash': 'Temporarily saves changes that are not yet committed, allowing you to switch branches without losing your current work',
        'git stash list': 'Look at files that have been stashed',
        'git stash apply <#>': 'Re-applies the changes from the specified stash to your working directory without removing them from the stash list',
        'git stash pop': 'Re-applies the most recent stash to your working directory and removes it from the stash list',
        "git stash drop stash@{n}": 'Remove a specific stash from stash list',
        'git clean -n': 'Shows the untracked files that would be removed if you were to run for real',
        'git clean -d': 'Will remove untracked files that are also in subdirectories',
        'git clean -f': 'Will actually get rid of untracked files in directory',
        'git remote add origin <url>': 'Set up remote connection to general GitHub repo. Name connection "origin"',
        'git push -u origin main': 'Set upstream so that you can push to remote. Need to do this before pushing. Git should give reminder if forgotten',
        'git push origin <branch>': 'Push a branch to remote GitHub connection named origin',
        'git push --all': 'Push all branches',
        'git fetch': 'Downloads updates from a remote repository, including new commits and branches, without merging or affecting your local branches',
        'git revert <hash>': 'Creates a new commit that undoes the changes introduced by the commit specified by a hash, without altering the commit history',
        'git clone <url>': 'Copy GitHub repo to a local machine',
        'git pull': 'Performs fetch and merge in one command',


        'lazy': 'lazy'
        }

        rundeck()
    

    print("Congrats! You've completed your study guide. Take it from the top?")
    studying=input().upper()

