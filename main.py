from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_bootstrap import Bootstrap
import numpy as np
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'DontTellAnyone'
Bootstrap(app)

dist = np.loadtxt('static/data.txt', delimiter=' ')
with open('static/node_names.txt', 'r') as f:
    node_names = [line.strip() for line in f.readlines()]
num_nodes = len(node_names)

def beautify(mask):
    s = ""
    for i in range(num_nodes):
        s += str(1 if (mask & (1<<i)) != 0 else 0)
    return s

def tsp_solver(selected_mask=(1<<num_nodes)-1, num_selected=num_nodes, start_node=-1):
    dp = np.full((1<<num_nodes, num_nodes), 1e9)

    if start_node == -1:
        start_node = 0
        for i in range(num_nodes):
            if selected_mask & (1<<i):
                start_node = i
                break

    dp[0][start_node] = 0
    for mask in range(1<<num_nodes):
        for next in range(num_nodes):
            if mask & (1<<next) == 0 and selected_mask & (1<<next):
                for cur in range(num_nodes):
                    if selected_mask & (1<<cur):
                        dp[mask | (1<<next)][next] = min(dp[mask | (1<<next)][next], dp[mask][cur] + dist[cur][next])

    # reconstruct the cycle:
    cur_mask = selected_mask
    order = []
    last = start_node
    for _ in range(num_selected-1, -1, -1):
        best_mid = -1
        for mid in range(0, num_nodes):
            if ((selected_mask & (1<<mid)) and \
                (cur_mask & (1<<mid)) != 0) and \
                (best_mid == -1 or dp[cur_mask][best_mid] + dist[best_mid][last] > dp[cur_mask][mid] + dist[mid][last]):
                best_mid = mid;
        order.append(best_mid)
        cur_mask ^= (1<<best_mid)
        last = best_mid

    # make the starting node the first node in the order
    while order[0] != start_node:
        order = order[1:] + [order[0]]

    return dp[selected_mask][start_node], order

def calc_mask(selected_nodes):
    mask = 0
    for i in selected_nodes:
        mask ^= 1 << i
    return mask

@app.route('/results', methods=['GET', 'POST'])
def results():
    shortest_dist = request.args['shortest_dist']
    ham_order = list(map(int, request.args['ham_order'].split('\t')))
    ham_cycle = [node_names[node] for node in ham_order] + [node_names[ham_order[0]]]

    return render_template('results.html', shortest_dist=shortest_dist, ham_cycle=ham_cycle)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        start_node = int(request.form['first_click'])
        selected_nodes = list(map(int, request.form.getlist("selected_nodes")))
        selected_mask = calc_mask(selected_nodes)

        res = tsp_solver(selected_mask, num_selected=len(selected_nodes), start_node=start_node)
        return redirect(url_for('results',
                                shortest_dist=res[0],
                                ham_order="\t".join(map(str,res[1]))))
    return render_template('index.html', node_names=node_names)

if __name__ == '__main__':
    app.run(debug=True)
