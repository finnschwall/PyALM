

# print(generate_doc('hist', test["data"]["hist"]))


# dic = {}
# def build_plugin_function_dict(plugin_entry, plugin_name):
#     global dic
#     print("------")
#     dic[plugin_name] = {}

#     ctx = click.Context(plugin_entry)
#     func_list = []
#     iter_dic = ctx.to_info_dict()["command"]["commands"]


#     for i in iter_dic:

#         if i == "help":
#             continue
#         par_prop = {}
#         par_args = []
#         par_kwargs = []
#         required = []
#         for par in iter_dic[i]["params"]:  # _name, par_info in iter_dic[i]["params"].items():
#             if par["name"] == "help":
#                 continue
#             entry = {}
#             conv_dic = {"Bool": "boolean", "Float": "number", "Int": "integer", "String": "string"}
#             par["type"]["param_type"] = conv_dic[par["type"]["param_type"]]
#             entry["type"] = par["type"]["param_type"]
#             par_prop[par["name"]] = {"type": par["type"]["param_type"]}
#             entry["name"] = par["name"]
#             if "help" in par:
#                 par_prop[par["name"]]["description"] = par["help"]
#                 entry["description"] = par["help"]
#             if par["required"]:
#                 required.append(par["name"])
#                 par_args.append(entry)
#             else:
#                 par_kwargs.append(entry)
#         params = {"type": "object", "properties": par_prop}
#         fun_exp = {"name": f"{plugin_name}__{i}", "parameters": params, "required": required}

#         fun_exp2 = {"args": par_args, "kwargs": par_kwargs}
#         if iter_dic[i]["help"]:
#             fun_exp["description"] = iter_dic[i]["help"]
#             fun_exp2["description"] = iter_dic[i]["help"]
#         func_list.append(fun_exp)

#         desired_order = ['description', 'args', 'kwargs']
#         ordered_dict = {key: fun_exp2[key] for key in desired_order if key in fun_exp2}

#         dic[plugin_name][i]=ordered_dict
#     pp(dic[plugin_name])
#     # print()
#     # print(func_list)
#     return func_list
