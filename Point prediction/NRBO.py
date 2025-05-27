# nrbo.py - 牛顿-拉夫森优化器的Python实现
import numpy as np

def initialization(nP, dim, ub, lb):
    """初始化搜索种群
    
    参数:
        nP: 种群大小
        dim: 问题维度
        ub: 上界(可以是标量或数组)
        lb: 下界(可以是标量或数组)
    
    返回:
        X: 初始化的种群，形状为(nP, dim)
    """
    # 处理标量边界情况
    if np.isscalar(lb) and np.isscalar(ub):
        lb = np.ones(dim) * lb
        ub = np.ones(dim) * ub
    
    # 确保边界是数组
    lb = np.array(lb)
    ub = np.array(ub)
    
    # 初始化种群
    X = np.zeros((nP, dim))
    for i in range(dim):
        X[:, i] = np.random.rand(nP) * (ub[i] - lb[i]) + lb[i]
    
    return X

def search_rule(best_pos, worst_pos, position, rho, flag=1):
    """牛顿-拉夫森搜索规则
    
    参数:
        best_pos: 最佳位置
        worst_pos: 最差位置
        position: 当前位置
        rho: 步长
        flag: 标志，控制搜索规则的应用方式
    
    返回:
        NRSR: 牛顿-拉夫森搜索步长
    """
    dim = len(position)
    # 计算Delta X
    del_x = np.random.rand(dim) * np.abs(best_pos - position)
    
    # 初始牛顿-拉夫森步骤
    denominator = 2 * (best_pos + worst_pos - 2 * position)
    # 避免除零
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    NRSR = np.random.randn() * ((best_pos - worst_pos) * del_x) / denominator
    
    # 根据标志调整位置
    if flag == 1:
        Xa = position - NRSR + rho
    else:
        Xa = best_pos - NRSR + rho
    
    # 进一步优化牛顿-拉夫森步骤
    r1, r2 = np.random.rand(), np.random.rand()
    yp = r1 * (np.mean([Xa, position], axis=0) + r1 * del_x)
    yq = r2 * (np.mean([Xa, position], axis=0) - r2 * del_x)
    
    # 避免除零
    denominator = 2 * (yp + yq - 2 * position)
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    NRSR = np.random.randn() * ((yp - yq) * del_x) / denominator
    
    return NRSR

def nrbo(fitness_func, dim, lb, ub, population=30, max_iter=100, df=0.6):
    """牛顿-拉夫森优化器
    
    参数:
        fitness_func: 目标函数，用于评估适应度
        dim: 问题维度
        lb: 下界(数组或标量)
        ub: 上界(数组或标量)
        population: 种群大小，默认30
        max_iter: 最大迭代次数，默认100
        df: 陷阱避免操作符决策因子，默认0.6
    
    返回:
        best_pos: 找到的最佳位置
        best_score: 最佳位置的适应度值
        convergence_curve: 收敛曲线
    """
    # 处理边界
    if np.isscalar(lb):
        lb = np.ones(dim) * lb
    if np.isscalar(ub):
        ub = np.ones(dim) * ub
    
    # 初始化种群
    position = initialization(population, dim, ub, lb)
    
    # 计算初始适应度
    fitness = np.zeros(population)
    for i in range(population):
        fitness[i] = fitness_func(position[i])
    
    # 找出最佳和最差个体
    ind = np.argsort(fitness)
    best_score = fitness[ind[0]]
    best_pos = position[ind[0]].copy()
    worst_score = fitness[ind[-1]]
    worst_pos = position[ind[-1]].copy()
    
    # 初始化收敛曲线
    convergence_curve = np.zeros(max_iter)
    
    # 主优化循环
    for it in range(max_iter):
        # 动态参数delta，随迭代减小
        delta = (1 - ((2 * (it+1)) / max_iter)) ** 5
        
        # 遍历所有个体
        for i in range(population):
            # 随机选择两个不同的个体进行差分进化
            a1, a2 = np.random.choice(population, 2, replace=False)
            
            # 计算步长rho
            rho = (np.random.rand() * (best_pos - position[i]) + 
                   np.random.rand() * (position[a1] - position[a2]))
            
            # 应用牛顿-拉夫森搜索规则
            flag = 1
            NRSR = search_rule(best_pos, worst_pos, position[i], rho, flag)
            X1 = position[i] - NRSR + rho
            X2 = best_pos - NRSR + rho
            
            # 更新个体位置
            Xupdate = np.zeros(dim)
            for j in range(dim):
                X3 = position[i, j] - delta * (X2[j] - X1[j])
                a1, a2 = np.random.rand(), np.random.rand()
                Xupdate[j] = a1 * (a1 * X1[j] + (1 - a2) * X2[j]) + (1 - a2) * X3
            
            # 陷阱避免操作符
            if np.random.rand() < df:
                theta1 = -1 + 2 * np.random.rand()
                theta2 = -0.5 + np.random.rand()
                beta = np.random.rand() < 0.5
                u1 = beta * 3 * np.random.rand() + (1 - beta)
                u2 = beta * np.random.rand() + (1 - beta)
                
                if u1 < 0.5:
                    X_TAO = (Xupdate + theta1 * (u1 * best_pos - u2 * position[i]) + 
                             theta2 * delta * (u1 * np.mean(position, axis=0) - u2 * position[i]))
                else:
                    X_TAO = (best_pos + theta1 * (u1 * best_pos - u2 * position[i]) + 
                             theta2 * delta * (u1 * np.mean(position, axis=0) - u2 * position[i]))
                
                Xnew = X_TAO
            else:
                Xnew = Xupdate
            
            # 执行边界检查
            Xnew = np.minimum(np.maximum(Xnew, lb), ub)
            
            # 评估新解
            Xnew_fitness = fitness_func(Xnew)
            
            # 更新最佳和最差位置
            if Xnew_fitness < fitness[i]:
                position[i] = Xnew.copy()
                fitness[i] = Xnew_fitness
                
                # 更新全局最佳解
                if fitness[i] < best_score:
                    best_pos = position[i].copy()
                    best_score = fitness[i]
            
            # 更新全局最差解
            if fitness[i] > worst_score:
                worst_pos = position[i].copy()
                worst_score = fitness[i]
        
        # 更新收敛曲线
        convergence_curve[it] = best_score
        
        # 打印迭代信息
        print(f'迭代 {it+1}: 最佳适应度 = {best_score:.8f}')
    
    return best_pos, best_score, convergence_curve