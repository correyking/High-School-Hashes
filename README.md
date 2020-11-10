# High-School-Hashes
#imports
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from pygam import LogisticGAM, s, f, te

#I'm surpressing warnings here because the PyGAM library warns you that the p-values are smaller than likely, which I am not concerned with.
import warnings
warnings.filterwarnings("ignore")

#load passing location data
df = pd.read_csv('https://raw.githubusercontent.com/ArrowheadAnalytics/next-gen-scrapy-2.0/master/pass_and_game_data.csv', low_memory=False)

#There's an additional index row we don't need, so I am getting rid of it here
df = df.iloc[0:,1:]
df.dropna(inplace=True)

#Function that will help get the data in the right shape every time I want to do this estimate
def kde_helper(df,name):
    '''Function to get data in the correct form for the KDE function
    inputs: dataframe, player name
    output: KDE applied to mesh grid, ready for plotting'''
    #Creating a mesh grid dividing each yard in half (so 4 units in a square yard),
    #between the boundaries of the x and y coordinates (the min and max of our data) supplied.
    m1 = df['x_coord'].loc[(df['name'].str.contains(name))]
    m2 = df['y_coord'].loc[(df['name'].str.contains(name))]
    #By using the same size grid each time I perform these estimates, I can make direct apples to apples comparisons between players. 
    #What I'm doing with this line is creating a "mesh grid" (think matrix) which I'll eventually evaluate the KDE on
    X, Y = np.mgrid[-30:30:121j, -10:60:141j]
    #flatten and stack these grids, giving a 2xn array of positions where n = 121*141 (the # of steps for each direction)
    #Basically what I am getting here is a "coordinate" for every single step I've created.
    #I start the x min at -30, so there will be 141 -30s - because -30 will be paired with every step I've created in the y direction.
    positions = np.vstack([X.ravel(), Y.ravel()])
    #Stack the values I care about in a 2xm array (basically transposing them here), where m is just the length of our supplied data
    values = np.vstack([m1, m2])
    #Perform the kernel estimation on the values I care about - you can think of this as "training" the kernel estimator
    kernel = stats.gaussian_kde(values)
    #Generate probabilities at the positions specified, transpose them, and put them back into the grid shape for plotting
    Z = np.reshape(kernel(positions).T, X.shape)
    return Z

   #Set our style
plt.style.use('seaborn-talk')

fig, ax1 =plt.subplots(1,1)

#This line is where the magic happens. Because of the way I performed the KDE, I have to rotate the data 270 degrees to plot in the orientation Iwant (np.rot90)
#Next, I want to make sure a pixel in the left direction is the same coordinate distance as a pixel in the vertical direction, so I set aspect to equal
#The extent is setting the coordinate system of the displayed image (along with the "origin" parameter). This is necessary to make sure that what we are indicating is the 20 yard line shows up as the 20 yardline in the pic
#Next, I want to normalize the colormap so that 0 is in the exact middle of the colormap. I can do this by having vmin and vmax have the same absolute value
#Lastly, I set the colormap parameter. I like "diverging" colormaps that have white in the middle for comparison plots, so it is clear which values are positive, negative, and 0.
plt.imshow(np.fliplr(np.rot90(diff_kde,3)),
           origin='lower', aspect='equal',
           extent=[-30, 30, -10, 60],
           norm = mpl.colors.Normalize(vmin=-0.0005, vmax=0.0005),
           cmap='RdBu_r')
#Add a "colorbar", a scale so people know what color represents what
cbar = plt.colorbar()
cbar.set_label("\nMahomes (Red) - Carr (Blue) passing densities")
#I don't really care about the values here, only the relative differences. 
#The values will change depending on how small I slice up our field. So, I only want to show the viewer what 0 is.
cbar.set_ticks([0])
#Set title, remove ticks and labels
ax1.set_title('Mahomes vs Carr - NFL Passing Densities')
ax1.set_xlabel('')
ax1.set_xticks([])

ax1.set_yticks([])

ax1.set_ylabel('')

#Remove any part of the plot that is out of bounds
ax1.set_xlim(-53.3333/2, 53.3333/2)

ax1.set_ylim(-10,60)

#Plot all of the field markings (line of scrimmage, hash marks, etc.)

for j in range(-10,60,1):
    ax1.annotate('--', (-8.1,j-0.5),
                 ha='right',fontsize =10)
    ax1.annotate('--', (8.1,j-0.5),
                 ha='left',fontsize =10)
for i in range(-10,60,5):
    ax1.axhline(i,c='k',ls='-',alpha=0.5, lw=1.5)
    
for i in range(-10,60,10):
    ax1.axhline(i,c='k',ls='-',alpha=0.7, lw=1.5)
    
for i in range(10,60-1,10):
    ax1.annotate(str(i), (-16.88,i-1.15),
            ha='right',fontsize =15,
                rotation=270)
    
    ax1.annotate(str(i), (16.88,i-0.65),
            ha='left',fontsize =15,
                rotation=90)    


ax1.annotate('Line of Scrimmage', (16,0),
             textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center',fontsize = 9) # horizontal alignment can be left, right or center


#I have to do a bit of cleaning to get the data in a form I can use for the model. First, I need to convert out pass_type column into a binary variable instead of the categorical complete, incomplete, touchdown, and interception. 
df['is_complete'] = 0
df.loc[((df['pass_type']=='COMPLETE') | (df['pass_type']=='TOUCHDOWN')), 'is_complete'] = 1

#Now let's see the distribution of our outcome
print(df.is_complete.mean())


#Get the features and outcomes I care about
X = df[['x_coord','y_coord']]
y = df[['is_complete']]
#Fit our model
gam = LogisticGAM().fit(X, y)
#Test the accuracy of our model
gam.summary()


def gam_helper(df):
    x = df[['x_coord','y_coord']]
    y = df['is_complete']
        #Similar to our KDE helper, I want a mesh grid that I will eventually evaluate the model on
    X, Y = np.mgrid[-30:30:121j, -10:60:141j]
        #Once again I want to flatten and stack our coordinates
    positions = np.vstack([X.ravel(), Y.ravel()])
        #Instead of a kde I fit a gam. Here I'm adjusting the number of splines to avoid overfitting, since we aren't doing any sort of hold out or cross validation in this post
    gam = LogisticGAM(s(0, n_splines=8) + s(1, n_splines=8) + te(0,1)).fit(x, y)
        #Generate probabilities at the positions specified, transpose them, and put them back into the grid shape for plotting
    Z = np.reshape(gam.predict_mu(positions.T).T, X.shape)
    return Z
    
#Call our function
pass_gam = gam_helper(df)


#Plot our output, same code as before
plt.style.use('seaborn-talk')

fig, ax1 =plt.subplots(1,1)

#This is where the magic happens here. Because of the way I performed the KDE, I have to rotate our data 270 degrees to plot in the orientation I want (np.rot90)
#Next, I want to make sure a pixel in the left direction is the same coordinate distance as a pixel in the vertical direction, so I set aspect to equal
#The extent is setting the coordinate system of the displayed image (along with the "origin" parameter). This is necessary to make sure that what I are indicating is the 20 yard line shows up as the 20 yardline in the pic
#Next, I want to normalize our colormap so that 0 is in the exact middle of the colormap. I can do this by having vmin and vmax have the same absolute value
#Lastly, I set the colormap parameter. I like "diverging" colormaps that have white in the middle for comparison plots, so it is clear which values are positive, negative, and 0.
plt.imshow(np.fliplr(np.rot90(pass_gam,3)),
           origin='lower', aspect='equal',
           extent=[-30, 30, -10, 60],
           norm = mpl.colors.Normalize(vmin=0, vmax=1),
           cmap='PiYG')
#Add a "colorbar", a scale so people know what color represents what
cbar = plt.colorbar()
cbar.set_label("\nEstimated Completion Probability")

#Set title, remove ticks and labels
ax1.set_title('League-wide Estimated Completion Probability')
ax1.set_xlabel('')
ax1.set_xticks([])

ax1.set_yticks([])

ax1.set_ylabel('')

#Remove any part of the plot that is out of bounds
ax1.set_xlim(-53.3333/2, 53.3333/2)

ax1.set_ylim(-10,60)

#Plot all of the field markings (line of scrimmage, hash marks, etc.)

for j in range(-10,60,1):
    ax1.annotate('--', (-8.1,j-0.5),
                 ha='right',fontsize =10)
    ax1.annotate('--', (8.1,j-0.5),
                 ha='left',fontsize =10)
for i in range(-10,60,5):
    ax1.axhline(i,c='k',ls='-',alpha=0.5, lw=1.5)
    
for i in range(-10,60,10):
    ax1.axhline(i,c='k',ls='-',alpha=0.7, lw=1.5)
    
for i in range(10,60-1,10):
    ax1.annotate(str(i), (-16.88,i-1.15),
            ha='right',fontsize =15,
                rotation=270)
    
    ax1.annotate(str(i), (16.88,i-0.65),
            ha='left',fontsize =15,
                rotation=90)    


ax1.annotate('Line of Scrimmage', (16,0),
             textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center',fontsize = 9) # horizontal alignment can be left, right or center
