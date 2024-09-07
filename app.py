import streamlit as st
from os import environ
from langchain.chains.conversation.memory import ConversationBufferMemory
 
from langchain.chains import LLMChain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from tespy.networks import Network
from tespy.components import DiabaticCombustionChamber, Turbine, Source, Sink, Compressor
from tespy.connections import Connection, Ref, Bus

# Load environment variables
load_dotenv()

# Set the OpenAI API key
api_key = environ.get('OPENAI_API_KEY')
environ["OPENAI_API_KEY"] = api_key

# Streamlit app configuration
st.set_page_config(page_title="Engineering Input Bot", page_icon=":robot_face:")

# Template for the assistant
template = """
You are a helpful assistant. Help the user with their requests.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
    ]
)

# Store session history in-memory (or in session_state for streamlit)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Initialize the chat session
session_id = "bcd1"
llm = ChatOpenAI(model="gpt-4o")

chain = prompt_template | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history"
)

# Container for chat history and responses
responses = {}

# Initial prompt to gather all required values
initial_prompt = """
You need to gather the following values from the user one at a time:
1. Compressor Efficiency (in %): Efficiency affects the pressure increase during compression.
2. Turbine Inlet Temperature (in °C): This influences the power generation in the turbine.
3. Combustion Chamber Thermal Input (in MW): This is the heat energy supplied during combustion.
4. Pressure Ratio for Compressor: The ratio of outlet to inlet pressure affects compression efficiency.
5. Fuel Composition (mass fraction): 
    a. Methane (CH4)
    b. Hydrogen (H2)
    c. Carbon Dioxide (CO2). 
    
    get these values and save in this format  CH4=0.8,H2=0.1,CO2=0.1
6. Oxygen Fraction in Flue Gas (in %): This affects combustion efficiency.

Ask for these values one by one from the user.If the values are not in range then prompt again, Below is the code which will be used to plt the graph later

# %%[sec_1]
from tespy.networks import Network
from tespy.components import (
    DiabaticCombustionChamber, Turbine, Source, Sink, Compressor
)
from tespy.connections import Connection, Ref, Bus

# define full fluid list for the network"s variable space
nw = Network(p_unit="bar", T_unit="C")

cp = Compressor("Compressor")
cc = DiabaticCombustionChamber("combustion chamber")
tu = Turbine("turbine")
air = Source("air source")
fuel = Source("fuel source")
fg = Sink("flue gas sink")
# %%[sec_2]
c2 = Connection(air, "out1", cc, "in1", label="2")
c3 = Connection(cc, "out1", fg, "in1", label="3")
c5 = Connection(fuel, "out1", cc, "in2", label="5")
nw.add_conns(c2, c3, c5)
# %%[sec_3]
cc.set_attr(pr=1, eta=1, lamb=1.5, ti=10e6)

c2.set_attr(
    p=1, T=20,
    fluid={"Ar": 0.0129, "N2": 0.7553, "CO2": 0.0004, "O2": 0.2314}
)
c5.set_attr(p=1, T=20, fluid={"CO2": 0.04, "CH4": 0.96, "H2": 0})

nw.solve(mode="design")
nw.print_results()
# %%[sec_4]
cc.set_attr(ti=None)
c5.set_attr(m=1)
nw.solve(mode="design")
# %%[sec_5]
cc.set_attr(lamb=None)
c3.set_attr(T=1400)
nw.solve(mode="design")
# %%[sec_6]
c5.set_attr(fluid={"CO2": 0.03, "CH4": 0.92, "H2": 0.05})
nw.solve(mode="design")
# %%[sec_7]
print(nw.results["Connection"])
# %%[sec_8]
nw.del_conns(c2, c3)
c1 = Connection(air, "out1", cp, "in1", label="1")
c2 = Connection(cp, "out1", cc, "in1", label="2")
c3 = Connection(cc, "out1", tu, "in1", label="3")
c4 = Connection(tu, "out1", fg, "in1", label="4")
nw.add_conns(c1, c2, c3, c4)

generator = Bus("generator")
generator.add_comps(
    {"comp": tu, "char": 0.98, "base": "component"},
    {"comp": cp, "char": 0.98, "base": "bus"},
)
nw.add_busses(generator)
# %%[sec_9]
cp.set_attr(eta_s=0.85, pr=15)
tu.set_attr(eta_s=0.90)
c1.set_attr(
    p=1, T=20,
    fluid={"Ar": 0.0129, "N2": 0.7553, "CO2": 0.0004, "O2": 0.2314}
)
c3.set_attr(m=30)
c4.set_attr(p=Ref(c1, 1, 0))
nw.solve("design")
c3.set_attr(m=None, T=1200)
nw.solve("design")
nw.print_results()
# %%[sec_10]
# unset the value, set Referenced value instead
c5.set_attr(p=None)
c5.set_attr(p=Ref(c2, 1.05, 0))
nw.solve("design")
# %%[sec_11]
cc.set_attr(pr=0.97, eta=0.98)
nw.set_attr(iterinfo=False)
import matplotlib.pyplot as plt
import numpy as np

# make text reasonably sized
plt.rc('font', **{'size': 18})

data = {
    'T_3': np.linspace(900, 1400, 11),
    'pr': np.linspace(10, 30, 11)
}
power = {
    'T_3': [],
    'pr': []
}
eta = {
    'T_3': [],
    'pr': []
}

for T in data['T_3']:
    c3.set_attr(T=T)
    nw.solve('design')
    power['T_3'] += [abs(generator.P.val) / 1e6]
    eta['T_3'] += [abs(generator.P.val) / cc.ti.val * 100]

# reset to base value
c3.set_attr(T=1200)

for pr in data['pr']:
    cp.set_attr(pr=pr)
    nw.solve('design')
    power['pr'] += [abs(generator.P.val) / 1e6]
    eta['pr'] += [abs(generator.P.val) / cc.ti.val * 100]

# reset to base value
cp.set_attr(pr=15)

fig, ax = plt.subplots(2, 2, figsize=(16, 8), sharex='col', sharey='row')

ax = ax.flatten()
[(a.grid(), a.set_axisbelow(True)) for a in ax]

i = 0
for key in data:
    ax[i].scatter(data[key], eta[key], s=100, color="#1f567d")
    ax[i + 2].scatter(data[key], power[key], s=100, color="#18a999")
    i += 1

ax[0].set_ylabel('Efficiency in %')
ax[2].set_ylabel('Power in MW')
ax[2].set_xlabel('Turbine inlet temperature °C')
ax[3].set_xlabel('Compressure pressure ratio')

plt.tight_layout()
fig.savefig('gas_turbine_parametric.svg')
plt.close()
# %%[sec_12]
c3.set_attr(T=None)

data = np.linspace(0.1, 0.2, 6)
T3 = []

for oxy in data[::-1]:
    c3.set_attr(fluid={"O2": oxy})
    nw.solve('design')
    T3 += [c3.T.val]

T3 = T3[::-1]

# reset to base value
c3.fluid.is_set.remove("O2")
c3.set_attr(T=1200)

fig, ax = plt.subplots(1, figsize=(16, 8))

ax.scatter(data * 100, T3, s=100, color="#1f567d")
ax.grid()
ax.set_axisbelow(True)

ax.set_ylabel('Turbine inlet temperature in °C')
ax.set_xlabel('Oxygen mass fraction in flue gas in %')

plt.tight_layout()
fig.savefig('gas_turbine_oxygen.svg')
plt.close()

# %%[sec_13]
# fix mass fractions of all potential fluids except combustion gases
c5.set_attr(fluid={"CO2": 0.03, "O2": 0, "H2O": 0, "Ar": 0, "N2": 0, "CH4": None, "H2": None})
c5.set_attr(fluid_balance=True)


data = np.linspace(50, 60, 11)

CH4 = []
H2 = []

for ti in data:
    cc.set_attr(ti=ti * 1e6)
    nw.solve('design')
    CH4 += [c5.fluid.val["CH4"] * 100]
    H2 += [c5.fluid.val["H2"] * 100]

nw._convergence_check()

fig, ax = plt.subplots(1, figsize=(16, 8))

ax.scatter(data, CH4, s=100, color="#1f567d", label="CH4 mass fraction")
ax.scatter(data, H2, s=100, color="#18a999", label="H2 mass fraction")
ax.grid()
ax.set_axisbelow(True)
ax.legend()

ax.set_ylabel('Mass fraction of the fuel in %')
ax.set_xlabel('Thermal input in MW')
ax.set_ybound([0, 100])

plt.tight_layout()
fig.savefig('gas_turbine_fuel_composition.svg')
plt.close()


Please proceed


"""

 
current_prompt = ''
# Display the app title
st.title("Engineering Input Bot")

# Initialize the conversation with the assistant
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    response = chain_with_history.invoke({"query": initial_prompt}, config={"configurable": {"session_id": session_id}})
    st.session_state['chat_history'] = [(initial_prompt, response.content)]
    st.session_state['responses'] = {}
    st.session_state['step'] = 1  # Set the initial step

# # Display the conversation history
# for user_input, bot_response in st.session_state['chat_history']:
#     st.write(f"**You**: {user_input}")
#     st.write(f"**Bot**: {bot_response}")

# Get the current step
steps = [
    "Compressor Efficiency (in %)",
    "Turbine Inlet Temperature (in °C)",
    "Combustion Chamber Thermal Input (in MW)",
    "Pressure Ratio for Compressor",
    "Fuel Composition (CH4, H2, CO2)",
    "Oxygen Fraction in Flue Gas (in %)"
]

# Ensure that the current step is within bounds
current_step = st.session_state['step']
if current_step <= len(steps):
    current_prompt = steps[current_step - 1]
else:
    st.write("All steps have been completed!")
    #plot_results(st.session_state['responses'])
    #st.stop()  # Stop further execution if all steps are done
    #st.write("### Collected Values:")
    #for key, value in st.session_state['responses'].items():
    #    st.write(f"{key}: {value}")
    #st.rerun()
    
    current_step = 1
    st.session_state['step'] = 1

    

# Text input area for user input
user_input = st.text_input(f"Please provide {current_prompt}", key='user_input', value="")

# When the user submits a response
if st.button('Send') and user_input:
    # Validate and store the input
    try:
        if current_step == 1:  # Compressor Efficiency
            efficiency = float(user_input)
            if 0 <= efficiency <= 100:
                st.session_state['responses']['compressor_efficiency'] = efficiency
            else:
                st.error("Please provide a value between 0 and 100 for Compressor Efficiency.")
                st.stop()

        elif current_step == 2:  # Turbine Inlet Temperature
            temperature = float(user_input)
            st.session_state['responses']['turbine_inlet_temperature'] = temperature

        elif current_step == 3:  # Combustion Chamber Thermal Input
            thermal_input = float(user_input)
            st.session_state['responses']['combustion_chamber_thermal_input'] = thermal_input

        elif current_step == 4:  # Pressure Ratio
            pressure_ratio = float(user_input)
            st.session_state['responses']['pressure_ratio'] = pressure_ratio

        elif current_step == 5:  # Fuel Composition
            # Validation for correct format
            if all('=' in item for item in user_input.split(',')):
                fuel_composition = user_input  # Expected in format: CH4=0.8,H2=0.1,CO2=0.1
                st.session_state['responses']['fuel_composition'] = fuel_composition
            else:
                st.error("Fuel composition format is incorrect. Ensure it is in the format: CH4=0.8,H2=0.1,CO2=0.1")
                st.stop()

        elif current_step == 6:  # Oxygen Fraction
            oxygen_fraction = float(user_input)
            if 0 <= oxygen_fraction <= 100:
                st.session_state['responses']['oxygen_fraction'] = oxygen_fraction
            else:
                st.error("Please provide a value between 0 and 100 for Oxygen Fraction.")
                st.stop()

        # Move to the next step
        st.session_state['step'] += 1

    except ValueError:
        st.error(f"Please provide a valid number for {current_prompt}.")
    
    # Check if the last step is reached
    if st.session_state['step'] > len(steps):
        st.success("All values have been successfully collected!")
        #plot_results(st.session_state['responses'])

    # Clear the input field by rerunning the app
    st.rerun()

# Display the collected responses so far
st.write("### Collected Values:")
for key, value in st.session_state['responses'].items():
    st.write(f"{key}: {value}")
