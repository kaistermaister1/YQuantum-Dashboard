import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Design System Colors
COLOR_QAOA = '#0066CC'
COLOR_DQI = '#00356B'
COLOR_CLASSICAL = '#2D8C3C'
COLOR_TRAVELERS_RED = '#E31837'

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#F8F9FA' # light gray app shell
plt.rcParams['figure.facecolor'] = '#FFFFFF'

def generate_plots():
    df = pd.read_csv('SOLUTIONS/HEURISTICS/run_summaries.csv')
    
    # Filter out rows where n_total is missing
    df = df.dropna(subset=['n_total'])
    
    # Calculate Approximation Ratio
    df['approx_ratio'] = df['best_profit'] / df['classical_opt_profit']
    
    # Calculate Feasibility Rate
    df['feasibility_rate'] = df['num_samples_feasible'] / df['num_samples_total']
    
    # Create output directory
    os.makedirs('public/plots/heuristics', exist_ok=True)
    
    # 1. Approximation Ratio vs Problem Size
    plt.figure(figsize=(8, 5))
    
    # Plot Classical
    classical_df = df[df['algorithm'] == 'classical']
    if not classical_df.empty:
        sns.lineplot(data=classical_df, x='n_total', y='approx_ratio', marker='o', label='Classical', color=COLOR_CLASSICAL, errorbar=None)
    
    # Plot QAOA COBYLA
    cobyla_df = df[(df['algorithm'] == 'qaoa') & (df['optimizer'] == 'cobyla')]
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='n_total', y='approx_ratio', marker='s', label='QAOA (COBYLA)', color=COLOR_QAOA, errorbar=('ci', 95))
        
    # Plot QAOA SPSA
    spsa_df = df[(df['algorithm'] == 'qaoa') & (df['optimizer'] == 'spsa')]
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='n_total', y='approx_ratio', marker='^', label='QAOA (SPSA)', color=COLOR_TRAVELERS_RED, errorbar=('ci', 95))
        
    plt.title('Solution Quality: Approximation Ratio vs Problem Size', fontsize=14, pad=15)
    plt.xlabel('Problem Size (n_total)', fontsize=12)
    plt.ylabel('Approximation Ratio', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Algorithm / Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/approx_ratio_vs_size.png', dpi=300)
    plt.close()
    
    # 2. Runtime vs Problem Size
    plt.figure(figsize=(8, 5))
    
    if not classical_df.empty:
        sns.lineplot(data=classical_df, x='n_total', y='runtime_sec', marker='o', label='Classical', color=COLOR_CLASSICAL, errorbar=None)
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='n_total', y='runtime_sec', marker='s', label='QAOA (COBYLA)', color=COLOR_QAOA, errorbar=('ci', 95))
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='n_total', y='runtime_sec', marker='^', label='QAOA (SPSA)', color=COLOR_TRAVELERS_RED, errorbar=('ci', 95))
        
    plt.title('Workflow Cost: Runtime vs Problem Size', fontsize=14, pad=15)
    plt.xlabel('Problem Size (n_total)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Algorithm / Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/runtime_vs_size.png', dpi=300)
    plt.close()
    
    # 3. Feasibility Rate vs Problem Size
    plt.figure(figsize=(8, 5))
    
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='n_total', y='feasibility_rate', marker='s', label='QAOA (COBYLA)', color=COLOR_QAOA, errorbar=('ci', 95))
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='n_total', y='feasibility_rate', marker='^', label='QAOA (SPSA)', color=COLOR_TRAVELERS_RED, errorbar=('ci', 95))
        
    plt.title('Constraint Handling: Feasibility Rate vs Problem Size', fontsize=14, pad=15)
    plt.xlabel('Problem Size (n_total)', fontsize=12)
    plt.ylabel('Feasibility Rate', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Algorithm / Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/feasibility_vs_size.png', dpi=300)
    plt.close()

    # 4. Circuit Resources vs Problem Size
    plt.figure(figsize=(8, 5))
    
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='n_total', y='num_qubits', marker='s', label='Qubits (COBYLA)', color=COLOR_QAOA, linestyle='-', errorbar=None)
        sns.lineplot(data=cobyla_df, x='n_total', y='two_qubit_gate_count', marker='s', label='Two-Qubit Gates (COBYLA)', color=COLOR_QAOA, linestyle='--', errorbar=None)
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='n_total', y='num_qubits', marker='^', label='Qubits (SPSA)', color=COLOR_TRAVELERS_RED, linestyle='-', errorbar=None)
        sns.lineplot(data=spsa_df, x='n_total', y='two_qubit_gate_count', marker='^', label='Two-Qubit Gates (SPSA)', color=COLOR_TRAVELERS_RED, linestyle='--', errorbar=None)
        
    plt.title('Hardware Resources: Qubits & Gates vs Problem Size', fontsize=14, pad=15)
    plt.xlabel('Problem Size (n_total)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Resource / Optimizer', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/resources_vs_size.png', dpi=300)
    plt.close()
    
    # 5. Approximation Ratio vs Circuit Depth (p)
    plt.figure(figsize=(8, 5))
    
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='p', y='approx_ratio', marker='s', label='QAOA (COBYLA)', color=COLOR_QAOA, errorbar=('ci', 95))
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='p', y='approx_ratio', marker='^', label='QAOA (SPSA)', color=COLOR_TRAVELERS_RED, errorbar=('ci', 95))
        
    plt.title('Depth Tradeoff: Approximation Ratio vs Circuit Depth (p)', fontsize=14, pad=15)
    plt.xlabel('Circuit Depth (p)', fontsize=12)
    plt.ylabel('Approximation Ratio', fontsize=12)
    plt.xticks([1, 2, 3])
    plt.ylim(0, 1.1)
    plt.legend(title='Algorithm / Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/approx_ratio_vs_p.png', dpi=300)
    plt.close()
    
    # 6. Objective Evaluations vs Problem Size
    plt.figure(figsize=(8, 5))
    
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='n_total', y='num_objective_evals', marker='s', label='QAOA (COBYLA)', color=COLOR_QAOA, errorbar=('ci', 95))
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='n_total', y='num_objective_evals', marker='^', label='QAOA (SPSA)', color=COLOR_TRAVELERS_RED, errorbar=('ci', 95))
        
    plt.title('Optimization Effort: Objective Evaluations vs Problem Size', fontsize=14, pad=15)
    plt.xlabel('Problem Size (n_total)', fontsize=12)
    plt.ylabel('Objective Evaluations', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Algorithm / Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/evals_vs_size.png', dpi=300)
    plt.close()

    # 7. Runtime vs N_local for fixed M_blocks
    plt.figure(figsize=(8, 5))
    qaoa_df = df[df['algorithm'] == 'qaoa']
    if not qaoa_df.empty:
        sns.lineplot(data=qaoa_df, x='N_local', y='runtime_sec', hue='M_blocks', marker='o', palette='viridis', errorbar=None)
        
    plt.title('Runtime Scaling for Fixed Number of Packages (M)', fontsize=14, pad=15)
    plt.xlabel('Coverage per Package (N_local)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Packages (M_blocks)')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/runtime_fixed_m.png', dpi=300)
    plt.close()
    
    # 8. Per-Block Qubits and Gates vs N_local
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.lineplot(data=qaoa_df, x='N_local', y='num_qubits', marker='o', label='Qubits', color=COLOR_QAOA, errorbar=None)
        sns.lineplot(data=qaoa_df, x='N_local', y='two_qubit_gate_count', marker='s', label='Two-Qubit Gates', color=COLOR_TRAVELERS_RED, errorbar=None)
        
    plt.title('Per-Block Hardware Resources vs Block Size (N_local)', fontsize=14, pad=15)
    plt.xlabel('Coverage per Package (N_local)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Resource')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/resources_vs_nlocal.png', dpi=300)
    plt.close()
    
    # 9. Runtime vs M_blocks for fixed N_local
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.lineplot(data=qaoa_df, x='M_blocks', y='runtime_sec', hue='N_local', marker='o', palette='plasma', errorbar=None)
        
    plt.title('Runtime Scaling for Fixed Coverage per Package (N_local)', fontsize=14, pad=15)
    plt.xlabel('Packages (M_blocks)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Coverage (N_local)')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/runtime_fixed_n.png', dpi=300)
    plt.close()

    # 10. 3D Runtime Landscape (Smooth Surface)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use SPSA as the representative for the smooth surface to avoid overlapping chaotic surfaces
    spsa_df = qaoa_df[qaoa_df['optimizer'] == 'spsa'].groupby(['N_local', 'M_blocks'])['runtime_sec'].mean().reset_index()
    
    if len(spsa_df) >= 3: # Need at least 3 points for a surface
        try:
            surf = ax.plot_trisurf(spsa_df['N_local'], spsa_df['M_blocks'], spsa_df['runtime_sec'], 
                                   cmap='plasma', edgecolor='none', alpha=0.9)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Runtime (s)')
        except Exception as e:
            print(f"Could not plot trisurf for runtime: {e}")
            ax.scatter(spsa_df['N_local'], spsa_df['M_blocks'], spsa_df['runtime_sec'], c=COLOR_TRAVELERS_RED, s=80)
                   
    ax.set_xlabel('Coverage per Package (N_local)', fontsize=10, labelpad=10)
    ax.set_ylabel('Packages (M_blocks)', fontsize=10, labelpad=10)
    ax.set_zlabel('Runtime (seconds)', fontsize=10, labelpad=10)
    ax.set_title('3D Runtime Landscape (QAOA SPSA)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/3d_runtime_surface.png', dpi=300)
    plt.close()
    
    # 11. 3D Approximation Ratio Landscape (Smooth Surface)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    spsa_approx_df = qaoa_df[qaoa_df['optimizer'] == 'spsa'].groupby(['N_local', 'M_blocks'])['approx_ratio'].mean().reset_index()
    
    if len(spsa_approx_df) >= 3:
        try:
            surf = ax.plot_trisurf(spsa_approx_df['N_local'], spsa_approx_df['M_blocks'], spsa_approx_df['approx_ratio'], 
                                   cmap='viridis', edgecolor='none', alpha=0.9)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Approximation Ratio')
        except Exception as e:
            print(f"Could not plot trisurf for approx ratio: {e}")
            ax.scatter(spsa_approx_df['N_local'], spsa_approx_df['M_blocks'], spsa_approx_df['approx_ratio'], c=COLOR_QAOA, s=80)
                   
    ax.set_xlabel('Coverage per Package (N_local)', fontsize=10, labelpad=10)
    ax.set_ylabel('Packages (M_blocks)', fontsize=10, labelpad=10)
    ax.set_zlabel('Approximation Ratio', fontsize=10, labelpad=10)
    ax.set_title('3D Solution Quality Landscape (QAOA SPSA)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/3d_approx_ratio_surface.png', dpi=300)
    plt.close()

    # 12. Profit Comparison (Scatter)
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.scatterplot(data=qaoa_df, x='classical_opt_profit', y='best_profit', hue='optimizer', style='optimizer', palette={'cobyla': COLOR_QAOA, 'spsa': COLOR_TRAVELERS_RED, 'random_batch': '#666666'}, s=100, alpha=0.7)
        max_val = max(qaoa_df['classical_opt_profit'].max(), qaoa_df['best_profit'].max())
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Optimal (y=x)')
    plt.title('QAOA Best Profit vs Classical Optimal', fontsize=14, pad=15)
    plt.xlabel('Classical Optimal Profit', fontsize=12)
    plt.ylabel('QAOA Best Profit', fontsize=12)
    plt.legend(title='Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/profit_scatter.png', dpi=300)
    plt.close()

    # 13. Approx Ratio Distribution by Optimizer
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.boxplot(data=qaoa_df, x='optimizer', y='approx_ratio', palette={'cobyla': COLOR_QAOA, 'spsa': COLOR_TRAVELERS_RED, 'random_batch': '#666666'})
    plt.title('Approximation Ratio Distribution by Optimizer', fontsize=14, pad=15)
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel('Approximation Ratio', fontsize=12)
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/approx_ratio_dist.png', dpi=300)
    plt.close()

    # 14. Runtime vs Objective Evals
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.scatterplot(data=qaoa_df, x='num_objective_evals', y='runtime_sec', hue='optimizer', style='optimizer', palette={'cobyla': COLOR_QAOA, 'spsa': COLOR_TRAVELERS_RED, 'random_batch': '#666666'}, s=100, alpha=0.7)
    plt.title('Runtime vs Objective Evaluations', fontsize=14, pad=15)
    plt.xlabel('Objective Evaluations', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(title='Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/runtime_vs_evals.png', dpi=300)
    plt.close()

    # 15. Feasibility vs Depth (p)
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.lineplot(data=qaoa_df, x='p', y='feasibility_rate', hue='optimizer', marker='o', palette={'cobyla': COLOR_QAOA, 'spsa': COLOR_TRAVELERS_RED, 'random_batch': '#666666'}, errorbar=('ci', 95))
    plt.title('Feasibility Rate vs Circuit Depth (p)', fontsize=14, pad=15)
    plt.xlabel('Circuit Depth (p)', fontsize=12)
    plt.ylabel('Feasibility Rate', fontsize=12)
    plt.xticks([1, 2, 3])
    plt.ylim(0, 1.1)
    plt.legend(title='Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/feasibility_vs_p.png', dpi=300)
    plt.close()

    # 16. Compiled Depth vs N_local
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.lineplot(data=qaoa_df, x='N_local', y='circuit_depth', hue='optimizer', marker='o', palette={'cobyla': COLOR_QAOA, 'spsa': COLOR_TRAVELERS_RED, 'random_batch': '#666666'}, errorbar=None)
    plt.title('Compiled Circuit Depth vs Block Size (N_local)', fontsize=14, pad=15)
    plt.xlabel('Coverage per Package (N_local)', fontsize=12)
    plt.ylabel('Compiled Circuit Depth', fontsize=12)
    plt.legend(title='Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/compiled_depth_vs_nlocal.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    generate_plots()
    print("Plots generated successfully.")
