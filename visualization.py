
import streamlit as st, pandas as pd, pydeck as pdk

_COL=[ [31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189],
       [140,86,75],[227,119,194],[127,127,127],[188,189,34],[23,190,207] ]
def _c(i): return _COL[i%len(_COL)]

def plot_network(stores,centers):
    st.subheader("Network Map")
    cen_df=pd.DataFrame(centers,columns=["Lon","Lat"])
    edges=[{"f":[r.Longitude,r.Latitude],
            "t":[cen_df.iloc[int(r.Warehouse)].Lon,cen_df.iloc[int(r.Warehouse)].Lat],
            "col":_c(int(r.Warehouse))+[120]} for r in stores.itertuples()]
    l_layer=pdk.Layer("LineLayer",edges,get_source_position="f",
                      get_target_position="t",get_color="col",get_width=2)
    cen_df[["r","g","b"]]=[_c(i) for i in range(len(cen_df))]
    wh_layer=pdk.Layer("ScatterplotLayer",cen_df,get_position="[Lon,Lat]",
                       get_fill_color="[r,g,b]",get_radius=35000,opacity=0.9)
    s_layer=pdk.Layer("ScatterplotLayer",stores,get_position="[Longitude,Latitude]",
                      get_fill_color="[0,128,255]",get_radius=12000,opacity=0.6)
    deck=pdk.Deck(layers=[l_layer,s_layer,wh_layer],
                  initial_view_state=pdk.ViewState(latitude=39,longitude=-98,zoom=3.5),
                  map_style="mapbox://styles/mapbox/light-v10")
    st.pydeck_chart(deck)

def summary(stores,total,out,in_,trans,wh,centers,demand,sqft_per_lb,
            rdc_on,consider_in,show_trans):
    st.subheader("Cost Summary")
    st.metric("Total annual cost",f"${total:,.0f}")
    cols=st.columns(4 if (consider_in or show_trans) else 2)
    i=0
    cols[i].metric("Outbound",f"${out:,.0f}"); i+=1
    if consider_in:
        cols[i].metric("Inbound",f"${in_:,.0f}"); i+=1
    if show_trans:
        cols[i].metric("Transfers",f"${trans:,.0f}"); i+=1
    cols[i].metric("Warehousing",f"${wh:,.0f}")

    df=pd.DataFrame(centers,columns=["Lon","Lat"])
    df["DemandLbs"]=demand
    df["SqFt"]=df["DemandLbs"]*sqft_per_lb
    st.subheader("Warehouse Demand & Size")
    st.dataframe(df[["DemandLbs","SqFt","Lat","Lon"]].style.format({"DemandLbs":"{:,}","SqFt":"{:,}"}))
