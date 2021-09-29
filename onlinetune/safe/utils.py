import numpy as np

def join_dtypes(*args):
    """
    Helper which joins dtypes d1 and d2, and returns a new dtype containing the fields of both d1 and d2.
    """
    fields = []
    for dtype in args:
        fields += [(f, dt[0]) for f, dt in dtype.fields.items()]

    return np.dtype(fields)

def join_dtype_arrays(a1, a2, target_dtype):
    """
    Initializes a new array with dtype target_dtype, and copies matching fields from a1 and a2 to the new array.
    """
    new_ar = np.zeros(shape=(), dtype=target_dtype)
    fields1 = a1 if isinstance(a1, dict) else a1.dtype.fields
    for f in fields1:
        if f in target_dtype.fields:
            new_ar[f] = a1[f]

    fields2 = a2 if isinstance(a2, dict) else a2.dtype.fields
    for f in fields2:
        if f in target_dtype.fields:
            new_ar[f] = a2[f]

    return new_ar


def maximize(f, X, mask=None, both=False, index=False):
    return minimize(lambda X: -f(X), X, mask.flatten(), both, index)

def minimize(f, X, mask=None, both=False, index_f=False):
    """
    Helper function for minimization.
    """
    if not both and not mask is None:
        X = X[mask]

    res = f(X)
    index = np.argmin(res)

    if both:
        X_masked = X[mask]
        res_masked = res[mask]
        index_masked = np.argmin(res_masked)
        if index_f:
            #print ("x:{}".format(X_masked[index_masked]))
            return X[index], res[index], X_masked[index_masked], res_masked[index_masked], list(res).index(res_masked[index_masked])


        return X[index], res[index], X_masked[index_masked], res_masked[index_masked]

    return X[index], res[index]


# helper functions for plotting
def plot_colored_region(axis, x1, x2, color):
    x1, x2 = np.asscalar(x1), np.asscalar(x2)
    trans = transforms.blended_transform_factory(axis.transData, axis.transAxes)
    l, u = 0., 0.03
    axis.fill_between([x1, x2], [l, l], [u, u], color=color, transform=trans, alpha=0.3)



def dimension_setting_helper(max_config, d):
    if max_config is None:
        return None

    if isinstance(max_config, int):
        return int(max_config)

    if isinstance(max_config, str):
        if max_config.startswith('d'):
            return d

        if '*' in max_config:
            factor, _ = max_config.split('*')
            return round(float(factor)*d)

    raise Exception("Invalid Config")



def plot_parameter_changes(axis, parameter_names, xold, xnew, l, u, tr_radius, x0):
    # normalize
    r = u - l
    xold_norm = (xold - l) / r
    x0_norm = (x0 - l) / r
    xnew_norm = (xnew - l) / r
    d = len(xold)

    # plot relative changes
    axis.set_ylim((0, 1))
    axis.set_xlim((-1, d))
    axis.set_ylabel('normalized parameter value')
    axis.set_xticks(range(d))

    if not parameter_names is None and len(parameter_names) > 0:
        axis.set_xticklabels(parameter_names, rotation='vertical')
    else:
        axis.set_xticklabels([f'X_{i}' for i in range(d)])
    w = 0.15

    # plot normalized changes
    for i, (x_start, x_stop, x_0i) in enumerate(zip(xold_norm, xnew_norm, x0_norm)):
        axis.plot([i - w, i + w], [x_start, x_start], color='C0')
        axis.plot([i - w, i + w], [x_stop, x_stop], color='C0')
        axis.plot([i - w, i + w], [x_0i, x_0i], color='C0', linestyle='--')

        # can't plot an arrow of zero length
        if np.abs(x_start - x_stop) < 0.0001:
            continue

        axis.arrow(i, x_start, 0, x_stop - x_start,
                   transform=axis.transData,
                   head_width=0.2,
                   head_length=0.05,
                   fc='C0', ec='C0',
                   length_includes_head=True, overhang=1, antialiased=True)

    axis.bar(range(d), np.abs(xnew - xold) / tr_radius, alpha=0.5, width=2 * w)


def plot_model_changes(axis, y_x0, y_xnew, std_xnew, y_coord):
    d = len(y_coord)

    twinx = axis.twinx()
    twinx.axhline(y_xnew - y_x0, color='C1')
    # twinx.axhline(ucb_xnew - y_x0)

    # axis.bar(range(d), ucb_coord - y_x0)
    twinx.bar(np.arange(d)+0.33, y_coord - y_x0, alpha=0.5, width=0.3, color='C1')

    twinx.set_ylabel('predicted objective increase')
    axis.set_title(f'Expected increase: {y_xnew - y_x0:.2} +- {std_xnew:.2}')


def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    m = int(m)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def locate(path):
    (modulename, classname) = path.rsplit('.', 1)

    m = __import__(modulename, globals(), locals(), [classname])
    if not hasattr(m, classname):
        raise ImportError(f'Could not locate "{path}".')
    return getattr(m, classname)


def split_int(i, p):
    split = []
    n = i / p  # min items per subsequence
    r = i % p  # remaindered items

    for i in range(p):
        split.append(int(n + (i < r)))

    return split




