import numpy as np
import pandas as pd

#creating a dataset 

data = pd.DataFrame()

#creating target variable

data['Gender'] = ['male','male','male','male','female','female','female','female']

#creating our feature variables

data['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
data['Weight'] = [180,190,170,165,100,150,130,150]
data['Foot_Size'] = [12,11,12,10,6,8,7,9]

#view the data

print("\n Dataset")
print("")
print(data)
 
#create an empty data frame

person = pd.DataFrame()

#creating some feature values for this single row 

person['Height'] = [5]
person['Weight'] = [130]
person['Foot_Size'] = [6]

#view the data 

print('\n Test Instance: ')
print(" ")
print(person)

n_male = data['Gender'][data['Gender'] == 'male'].count()
n_male

n_female = data['Gender'][data['Gender'] == 'female'].count()
n_female

#total rows

total_ppl = data['Gender'].count()
total_ppl

#no of males divided by the total rows 

p_male = n_male / total_ppl     #(4/8)
p_male

p_female = n_female / total_ppl     #(4/8)
p_female

# group the data by gender & calculate the means of each feature
# for eg - height = (6+5.92+5.58+5.92) / 4

data_means = data.groupby('Gender').mean() 
data_means

#calculate of mean

print('\n Dataset Mean')
print(" ")
print(data_means)

# calculate the data variance 
# variance = summation of((mean - x) ** 2) / n 

data_variance = data.groupby('Gender').var()
print(data_variance)
          
#mean for male 

male_height_mean = data_means['Height'][data_means.index == 'male'].values[0]

male_weight_mean = data_means['Weight'][data_means.index == 'male'].values[0]

male_footsize_mean  = data_means['Foot_Size'][data_means.index == 'male'].values[0]

print("male_height_mean: ", male_height_mean)
print("male_weight_mean: ", male_weight_mean)
print("male_footsize_mean: ", male_footsize_mean)

#variance for male 

male_height_variance = data_variance['Height'][data_variance.index == 'male'].values[0] 

male_weight_variance = data_variance['Weight'][data_variance.index == 'male'].values[0]

male_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'male'].values[0]

print("male_height_variance: ",male_height_variance)
print("male_weight_variance: ",male_weight_variance)
print("male_footsize_variance: ",male_footsize_variance)

# for female now
# mean for female

female_height_mean = data_means['Height'][data_means.index == 'female'].values[0]

female_weight_mean = data_means['Weight'][data_means.index == 'female'].values[0]

female_footsize_mean  = data_means['Foot_Size'][data_means.index == 'female'].values[0]

print("female_height_mean: ", female_height_mean)
print("female_weight_mean: ", female_weight_mean)
print("female_footsize_mean: ", female_footsize_mean)

#variance for female 

female_height_variance = data_variance['Height'][data_variance.index == 'female'].values[0] 

female_weight_variance = data_variance['Weight'][data_variance.index == 'female'].values[0]

female_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'female'].values[0]

print("female_height_variance: ",female_height_variance)
print("female_weight_variance: ",female_weight_variance)
print("female_footsize_variance: ",female_footsize_variance)

# create a function which calculates p(x|y)

def p_x_given_y(x,mean_y, variance_y):
    
    #input the arguments into a probability density function
    p = 1/(np.sqrt(2*np.pi*variance_y))* np.exp((-(x-mean_y) ** 2)/(2*variance_y))
    return p

# numerator of the posterior if the unclassified observation is a male 

print('\n Probability male: ')

prob_male = p_male*p_x_given_y(person['Height'][0],male_height_mean,male_height_variance)* p_x_given_y(person['Weight'][0],male_weight_mean,male_weight_variance)* p_x_given_y(person['Foot_Size'][0],male_footsize_mean,male_footsize_variance)

print(prob_male)

print('\n Probability female: ')

prob_female = p_female*p_x_given_y(person['Height'][0],female_height_mean,female_height_variance)* p_x_given_y(person['Weight'][0],female_weight_mean,female_weight_variance)* p_x_given_y(person['Foot_Size'][0],female_footsize_mean,female_footsize_variance)

print(prob_female)

if(prob_male > prob_female):
    print("target label: Male")
else:
    print("target label: Female")
