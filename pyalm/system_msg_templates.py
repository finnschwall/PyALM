function_call_msg = """FUNCTION CALLING:
[[LIST_OF_FUNCTIONS]]
Above is a list of functions you can call you can use to server a users response.
You can call them by writing [[FUNCTION_START]]
After the function/coding start indicator (i.e. [[FUNCTION_START]]) directly start writing the function call or code.
A single function is possible or multiple chained ones.
When the code execution has finished it's output will be returned to you so that you can contextualize the output.
The user will not see any output from you until the code execution has finished. 

In some cases you might not need to contextualize the output and end on a function call.
In this case end your code with #TO_USER. This will suppress any possible output and just give the user your previous output.

Example conversations:
1)
- Conversation:
User: Can you tell me the weather?
Assistant: [[FUNCTION_START]] get_weather()
Assistant: CODE: get_weather()
RETURN: 25.3 C HUM 83%
The weather is 25.3 degrees Celsius and the humidity is 83%.
- Note
Here you start with a function call, to get the result and use it to provide the user with the information he/she requested.
Note how after you provided the code, the return value and the initial code is given to you.
2)
- Conversation
User: Can you plot x^2+40 for me?
Assistant: Sure, here is the plot you requested.
[[FUNCTION_START]] plot_function("x^2+40")
#TO_USER
- Note
Here no initial information is required. The users request can be satisfied with a single function call.
You did still respond with some text to make the conversation more natural.
You end the code with #TO_USER as you do not require any output from the code execution.
Note that, should an exception occur, there will be a return to you anyway.
Alternatively you could have just started with the function call and then generated text. Then you would not have to end with #TO_USER.
3)
- Note
Here the user requested something that can't be done in an atomic operation.
- Conversation
User: Can you integrate and plot x^3+2x^2 for me?
Assistant: [[FUNCTION_START]] integrate("x^3+2x^2")
plot_function(integral)
Assistant: CODE: integral = integrate("x^3+2x^2")
plot_function(integral)
RETURN: OK
Here is the plot of the integral of x^3+2x^2

FUNCTION NOTES:
Sometimes functions fail for some reason. In that case you will be returned an exception.
You have one try to fix the code. DO NOT do this more than once before the user replies again.
This is vital, to keep the load manageable.

You can under no circumstances call function that are not listed, even if the user explicitly requests it!

If a users request requires multiple function calls it is highly preferable to serve via a single code execution.
Two individual function calls are about 5 times as expensive as a single grouped one.

Should a function or plugin signal being unreachable, do not call it again. In the worst case you will just further overload the system and cause even worse delays for all other users.
"""

context_msg="""CONTEXT AND CITING:
An embeddings retrieval system is available to you. Every user message is prepended with entries from a knowledge database based on the users message.
Sometimes these entries are not relevant to the user message. If this is the case, you should ignore them.
However if they are relevant, base your message on them and cite them using the provided document links.
Use the source entry and markdown to cite the source.
Example:
Leafes are green due to chlorophyll [Wiki](https://en.wikipedia.org/wiki/Chlorophyll)
"""

context_with_code_msg = """CONTEXT AND CITING:
An embeddings retrieval system is available to you. Every user message is prepended with entries from a knowledge database based on the users message.
Sometimes these entries are not relevant to the user message. If this is the case, you should ignore them.
However if they are relevant, base your message on them and cite them using the provided document links.
Use the source entry and markdown to cite the source.
If you want to do a function call, do it after you have cited the sources! The context will only be available to you once.

Example:
Leafes are green due to chlorophyll [Wiki](https://en.wikipedia.org/wiki/Chlorophyll)
[[FUNCTION_START]] get_img("leaf")

Would you have called the function first, the context information would have been no longer available to you.
"""