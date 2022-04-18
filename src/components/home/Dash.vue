<template>
  <v-row align="center" justify="center" class="py-5">
    <v-col align="center" justify="center" cols="8" v-if="!active_accidents">
      <v-alert
        color="primary"
        dark
        icon="mdi-eye"
        border="left"
        prominent
      >No active accidents at the moment.Watching for accidents...</v-alert>
    </v-col>
    <v-col align="center" justify="center" cols="8" v-if="active_accidents">
      <v-alert type="error">{{active_accidents}} Unattended accidents are present!</v-alert>
    </v-col>
    <v-col
      align="center"
      justify="center"
      cols="8"
      v-for="accident in latest_accidents"
      :key="accident.id"
    >
      <span :key="accident" v-if="!accident.is_dismissed && !accident.is_reported">
        <v-card max-width="1000" align="center" color="error" :key="accident" dark>
          <v-card-title class="title">
            <span>
              Location: {{accident.location.latitude}},{{accident.location.longitude}}
              <v-btn
                light
                @mouseover="accident.showMap=true"
                @mouseout="accident.showMap=false"
                @click="GoToUrl(accident.longitude,accident.latitude)"
              >
                <v-icon>mdi-map</v-icon>Directions
              </v-btn>
            </span>
            <v-spacer></v-spacer>
            <v-switch v-model="accident.showMap">
              <template v-slot:label>
                <strong style="color:white">Show in Map</strong>
              </template>
            </v-switch>
            <v-spacer></v-spacer>
            <span>Date:{{accident.date}}</span>
          </v-card-title>
          <!-- <v-hover v-slot:default="{ hover }"> -->
          <v-img max-height="500" :src="accident.img_src">
            <div v-if="accident.showMap">
              <Map :origin="coordinates" :destination="accident.location" />
            </div>
          </v-img>
          <!-- </v-hover> -->
          <Response :doc_id="accident.id" />
        </v-card>
      </span>
    </v-col>
  </v-row>
</template>

<script>
import db from "@/firebase/init";
import moment from "moment";
import firebase from "firebase";
import Response from "@/components/home/Response";
import Map from "@/components/home/Map";
export default {
  name: "Dash",
  components: {
    Response,
    Map,
  },

  data() {
    return {
      accidents: [],
      searchUrl: "https://www.google.com/maps/dir/?api=1&",
      coordinates: {
        latitude: null,
        longitude: null,
      },
      cred: this.$root.cred,
    };
  },
  computed: {
    latest_accidents() {
      return this.accidents.slice().reverse();
    },
    active_accidents() {
      return this.$root.accidents.filter((acc) => {
        return !acc.is_reported && !acc.is_dismissed;
      }).length;
    },
  },
  methods: {
    updateAccidents() {
      this.$root.accidents = this.accidents;
    },
    GoToUrl(longitude, latitude) {
      window.open(
        encodeURI(
          this.searchUrl +
            "origin=" +
            this.coordinates.latitude +
            "," +
            this.coordinates.longitude +
            "&destination=" +
            longitude +
            "," +
            latitude +
            "&travelmode=driving"
        )
      );
    },
  },
  created() {
    console.log(this.$root.username);
    let user = firebase.auth().currentUser;
    console.log(user.uid);
    let accident_collection = db.collection("accidents").orderBy("timestamp");
    let user_collection = db.collection("users").doc(this.cred.user.uid);
    //Set user location
    user_collection.get().then((doc) => {
      var location = doc.data().location;
      this.coordinates.latitude = location.latitude;
      this.coordinates.longitude = location.longitude;
      console.log(this.coordinates.longitude);
    });
    //Look for changes in accidents collection
    accident_collection.onSnapshot((snapshot) => {
      snapshot.docChanges().forEach((change) => {
        if (change.type == "added") {
          let doc = change.doc;
          this.accidents.push({
            id: doc.id,
            location: {
              latitude: doc.data().Latitude,
              longitude: doc.data().Longitude,
            },
            img_src: doc.data().URL,
            is_dismissed: doc.data().is_dismissed,
            is_reported: doc.data().is_reported,
            date: moment(doc.data().timestamp).format("lll"),
            showMap: false,
          });
          this.updateAccidents();
        }
        if (change.type == "modified") {
          // let doc = change.doc;
          //pop all accidents
          while (this.accidents.length > 0) {
            this.accidents.pop();
          }
          //get all items again
          firebase
            .firestore()
            .collection("accidents")
            .get()
            .then((snapshot) => {
              snapshot.forEach((doc) => {
                this.accidents.push({
                  id: doc.id,
                  location: {
                    longitude: doc.data().Longitude,
                    latitude: doc.data().Latitude,
                  },
                  img_src: doc.data().URL,
                  is_dismissed: doc.data().is_dismissed,
                  is_reported: doc.data().is_reported,
                  date: moment(doc.data().timestamp).format("lll"),
                  showMap: false,
                });
              });
            });
          this.updateAccidents();
        }
      });
    });
  },
};
</script>
<style scoped>
.v-card--reveal {
  align-items: center;
  bottom: 0;
  justify-content: center;
  opacity: 0.5;
  position: absolute;
  width: 100%;
}
</style>
