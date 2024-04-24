#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

class TreeNode {
public:
    int val;
    vector<TreeNode*> children;

    TreeNode(int v) : val(v) {}
};

class Tree {
public:
    TreeNode* root;

    Tree(TreeNode* r) : root(r) {}
};

// Function to construct a tree from user input
Tree constructTree() {
    int n; // Number of nodes in the tree
    cout << "Enter the number of nodes in the tree: ";
    cin >> n;

    vector<TreeNode*> nodes(n);
    for (int i = 0; i < n; ++i) {
        nodes[i] = new TreeNode(i); // Create a new node with value i
    }

    int parent, child;
    Tree tree(nodes[0]); // Assume the first node is the root
    cout << "Enter the parent-child relationships (parent child), -1 to end:" << endl;
    while (true) {
        cin >> parent;
        if (parent == -1) break;
        cin >> child;
        nodes[parent]->children.push_back(nodes[child]);
    }

    return tree;
}

// Parallel Breadth First Search on Tree
void parallelBFS(Tree& tree) {
    queue<TreeNode*> q;
    q.push(tree.root);

    while (!q.empty()) {
        int size = q.size();
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            TreeNode* current;
            #pragma omp critical
            {
                current = q.front();
                q.pop();
            }
            cout << "Visited: " << current->val << endl;

            for (TreeNode* child : current->children) {
                #pragma omp critical
                q.push(child);
            }
        }
    }
}

// Parallel Depth First Search on Tree
void parallelDFS(Tree& tree) {
    stack<TreeNode*> s;
    s.push(tree.root);

    while (!s.empty()) {
        TreeNode* current;
        #pragma omp critical
        {
            current = s.top();
            s.pop();
        }
        cout << "Visited: " << current->val << endl;

        for (TreeNode* child : current->children) {
            #pragma omp critical
            s.push(child);
        }
    }
}

int main() {
    // Construct the tree
    Tree tree = constructTree();

    // Perform parallel BFS
    cout << "\nParallel BFS on Tree:" << endl;
    parallelBFS(tree);

    // Perform parallel DFS
    cout << "\nParallel DFS on Tree:" << endl;
    parallelDFS(tree);

    return 0;
}
