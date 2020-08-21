import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import umap

header_font = fm.FontProperties(fname='../src/visualizations/Fonts/raleway/Raleway-Regular.ttf', size = 22)
text_font = fm.FontProperties(fname='../src/visualizations/Fonts/lato-1/Lato-Regular.ttf', size = 19)
plt.rcParams['legend.title_fontsize'] = 17 #Legend text size

def umap_transform(actual_vectors, predicted_vector):
    """Performs a UMAP transformation.

    Args:
        Actual_vectors (array): Secondary vectors to be plotted.
        Predicted_vector (array): Primary vector (i.e., input text vector)

    Returns:
        Array of 2-D UMAP transformed vectors.
    """
    all_vecs = np.append(actual_vectors, predicted_vector, axis = 0)
    transformed = umap.UMAP().fit_transform(all_vecs)
    return transformed

def umap_viz(vectors,
             pred_text,
             pred_color,
             plot_recs = False,
             df=None,
             dists=None,
             recs=None,
             recs_colors=None
            ):
    """Plots UMAP projection scatterplot.

    Much of the code is simple aesthetic parameters. Essentially,
    all potential vectors are plotted, the input text vector is plotted
    in 1 color, and the recommendations are plotted in specified colors
    of their own. This is to highlight the inputted text point, as well
    as the recommendations in the projection space.

    Vectors (array): UMAP vectors.
    Pred_text (str): Original inputted text -- used for legend.
    Pred_color (str): Color of text dot.
    Plot_recs (bool): Whether to plot recommendations or not. Default
      is false.
    Df (dataframe): Dataframe containing recommendation information.
      Default is false.
    Dists (array): Recommendation indices corresponding to dataframe.
      Default is false.
    Recs (int): Number of recommendations to plot. Default is none.
    Recs_colors (list): Colors to use for recommendations. There should
      be 1 color for each recommendation.

    Returns:
      Matplotlib scatterplot.
    """
    fig, ax = plt.subplots(figsize = (20,20))
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.set_facecolor('w')
    ax.set_facecolor('w')
    ax.set_title('Contextual Topic Embeddings - UMAP Projection', fontproperties = header_font)
    ax.scatter([coords[0] for coords in vectors], [coords[1] for coords in vectors], s = 5, color = 'grey', alpha = .03) # Plot all points
    inputted_text_plot = [ax.scatter(vectors[-1][0], vectors[-1][1], s = plt.rcParams['lines.markersize'] ** 2.5, color = pred_color, label = pred_text[:49]+'...')] # Plot predicted vector (i.e., user-inputted text)
    posts = []
    if plot_recs:
        for i in range(recs):
            post = ax.scatter(vectors[dists[i]][0], vectors[dists[i]][1], s = plt.rcParams['lines.markersize'] ** 2.5, color = recs_colors[i], label = df.loc[dists[i], 'title'])
            posts.append(post)
            posts_legend = ax.legend(posts,
                             [posts[i].get_label() for i in range(len(posts))],
                             loc='right',
                             title = 'Recommendations\n',
                             prop=text_font)
        ax.add_artist(posts_legend)
    inputted_text_legend = ax.legend(inputted_text_plot,
                                     [inputted_text_plot[i].get_label() for i in range(len(inputted_text_plot))],
                                     loc = 'upper right',
                                     title = 'Inputted Text\n',
                                     prop = text_font)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both',length=0)
    ax.add_artist(inputted_text_legend)
    return
