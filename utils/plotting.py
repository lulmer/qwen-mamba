import matplotlib.pyplot as plt 

# Plot the results
def plot_loss(token_counts, mean_iteration_losses, experiment_name):

    plt.figure(figsize=(10, 6))
    plt.plot(token_counts, mean_iteration_losses, 'b-', linewidth=2, label='Mean Attention L2 Loss')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Attention L2 Loss')
    plt.title('Training Progress: Mean Attention Loss vs Tokens Processed')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Format x-axis to show tokens in K/M
    def format_tokens(x, pos):
        if x >= 1_000_000:
            return f'{x/1_000_000:.1f}M'
        elif x >= 1_000:
            return f'{x/1_000:.1f}K'
        else:
            return f'{int(x)}'

    from matplotlib.ticker import FuncFormatter
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_tokens))

    # Save the plot
    plt.savefig(f"{experiment_name}_training_loss.png", dpi=300, bbox_inches='tight')
    plt.show()
