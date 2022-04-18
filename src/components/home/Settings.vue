<template>
  <div>
    <v-row align="center" justify="center">
      <v-col align="center">
        <v-card max-width="1000">
          <v-card-title>Change your location settings</v-card-title>
          <v-form
            class="pa-3"
            ref="form"
            v-model="valid"
            lazy-validation
            @submit.prevent="update_location()"
          >
            <v-text-field v-model="longitude" :rules="[locationRules]" label="Longitude"></v-text-field>
            <v-text-field v-model="latitude" :rules="[locationRules]" label="Latitude"></v-text-field>
            <v-btn :disabled="!valid" color="primary" class="mr-4" type="submit">Save</v-btn>
          </v-form>
        </v-card>
        <v-snackbar :right="true" :top="true" :timeout="2000" v-model="snackbar">
          <div>
            <v-icon dark>mdi-account-check</v-icon>Settings have been updated
          </div>
          <template v-slot:action="{ attrs }">
            <v-btn color="red" text v-bind="attrs" @click="snackbar = false">Close</v-btn>
          </template>
        </v-snackbar>
      </v-col>
    </v-row>
  </div>
</template>

<script>
import db from "@/firebase/init";
export default {
  name: "Settings",
  data: () => ({
    valid: true,
    snackbar: false,
    locationRules: (v) => {
      if (!isNaN(v)) return true;
      return "Enter a number";
    },
    longitude: 0,
    latitude: 0,
  }),
  methods: {
    update_location() {
      db.collection("users")
        .doc(this.$root.cred.user.uid)
        .update({
          location: { longitude: this.longitude, latitude: this.latitude },
        })
        .then(() => {
          this.snackbar = true;
        });
    },
  },
};
</script>