import random
import math

def rundeck():
    study_dict=tricky_dict

    rand_conf_keys=random.sample(list(confident_dict.keys()), math.ceil(len(confident_dict)*0.25))

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

subjlist=['PYTHON', 'R', 'APPLIED STATISTICS IN R', 'ASIR', 'PML', 'DS']

studying='YES'

while studying=='YES':

    subject=''
    print("Which subject would you like to study?")
    while subject not in subjlist:
        subject=input().upper()
        if subject=='QUIT':
            quit()
        if subject not in subjlist:
            print('Sorry, that subject is not available. Please select another.')

    if subject=='PYTHON':

        confident_dict={
        ".append()": "This method adds a single item to a list",
        ".extend()": "This method adds multiple items to a list",
        ".remove()": "This deletes an item from a list based on value",
        "del list[index]": "This deletes an item from a list based on index",
        "df.columns": "Creates a list of columns in a pandas data frame",
        "df['Column'].astype()": "Converts one data type to another for specified column(s). Specify a column in answer",
        "df.loc[]": "Used to access a group of rows and columns by label(s) or a boolean array. Let's you subset data frame using logicals.",
        "df.iloc[]": "Gets or sets value(s) at speicifed data frame indices",
        "df.describe()": "Provides summary statistics of numerical data columns",
        "df.sort_values()": "Sorts a data frame based on selected column(s)",
        "~": "Not operator in .loc method",
        ".add()": "Adds items to a set",
        "pd.DataFrame([('row1col1', 'row1col2'), ('row2col1', 'row2col2'), ('row3col1', 'row3col2')], columns=['col1','col2'])": "Create pandas dataframe using a list tuples. 3 rows 2 cols",
        "pd.DataFrame({'col1': ['row1', 'row2', 'row3'], 'col2': ['row1', 'row2', 'row3']})": "Create pandas data frame using dictionary of lists",
        ".sort()": "Sorts in place and returns nothing",
        "sorted()": "Creates a returns a new sorted sequence",
        "f'Hello {name}. You are studying {language}.'": "An f-string that would generate 'Hello Mac. You are studying Python.' with Mac and Python assigned as variables",
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
        'df.describe()': 'Provides summary statistics about a data frame',



        'lazy': 'lazy'
        }

        tricky_dict={
        "df.set_index(['Column'])": "Make it so a certain column is now the index",
        "df.sort_index(ascending=False)": "Sort by index by descending",
        "df['Column'].value_counts()": "Creates a frequency for a column. Specify column.",
        '.replace()': 'Replaces a specified string with another specified string',
        '.split()': 'Splits a string into a list',
        '.index()': 'Returns the index of a value from a list',
        'pd.concat([df1, df2])': 'Appends dataframes. Include df1 and df2 as the dataframes getting appended',
        'matplotlib': 'Foundational library for visualizations. Many other viz libraries are built on top of this one',
        'seaborn': 'A statistical viz library built on top of matplotlib',
        'statsmodels': 'A library with many advanced statistical functions',
        'scipy': 'Library that provides advanced scientific computing, including functions for optimization, linear algebra, image processing and much more',
        'scikit-learn': 'Most popular machine learning library for Python (not deep learning)',
        'df.corr()': 'Calculates correlations between columns',
        "pd.merge(df1, df2, left_on=['Column'], right_on=['Column'], how='left')": 'Left join two datasets. Specify dfs and columns to join on.',
        "df.drop_duplicates(subset=['Column'])": "Deduplicates a data frame by a column(s). Specify column.",
        'np.array([])': 'Create NumPy array',
        'array[[]]': 'Multi-index from array',
        "df['Column'].nunique": 'Number of unique rows in column',
        "pd.to_datetime(df['Column']).dt.year": 'Get year part of pandas column that is not already in a recognizable datetime format',
        'dropna=False': 'Argument in value_counts() that will include NaN values',
        "df.boxplot(column=['Column'])": "Creates a boxplot of a series",
        'normalize=True': 'Argument in value_counts() that gets percentage breakdown',
        'df.fillna()': "Will fill NA values in a column with a specified value",
        'np.where(condition, x, y)': 'NumPy funciton that returns values from x or y depending on condition',
        'df.dropna()': 'Drops NA values',
        'np.zeros(size, dtype)': 'Empty numpy array. Enter hypothetical args',
        'np.zeros_like()': 'Empty numpy array in the shape of another array',
        'np.linspace(min, max, n)': 'Linearly spaced numpy array that includes number of data points specified',
        'np.arange(min, max, interval)': 'Numpy array with equally spaced specified intervals',
        'np.random.random(size=(8,8))': 'Random 8x8 numpy array',
        '@': 'Numpy matrix multiplication',


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
        'Poisson': 'A discrete distribution describes discrete occurrences over a continuum or interval',
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


    if subject=='PML':

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
        'Instance based': 'Learning method where agent learns examples and then generalizes to new cases by using a similarity measure to compare',
        'Model based': 'Learning method where agent uses set of examples to build model and uses said model to make predictions',
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
        'Precision Recall Tradeoff': 'Graph that displays the inverse relationship between precision and recall',
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
        'Projection': 'Method of dimension reduction that simply flattens data down into a lower dimension. Like 2D to 3D',
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


    if subject=='DS':

        confident_dict={
        'DMBS': 'A generalized software system for manipulating databases',


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

    if subject in ['DC', 'DECISION ANALYSIS']:

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
        'Right hand side': 'The solution to an LP problem is degenerate if the ________ of any of the constraints have an allowable increase or an allowable decrease of 0'



        'lazy': 'lazy'
        }

        rundeck()



    print("Congrats! You've completed your study guide. Take it from the top?")
    studying=input().upper()
